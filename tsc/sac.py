"""Discrete Soft Actor-Critic (Christodoulou 2019; Zhou et al. 2024).

Ported from dqn/sac_agent.py of the original partial-detection code (same CNN encoders, twin
critics, Polyak targets). Changes:
  * optional automatic temperature: alpha is learned so that the expected
    policy entropy tracks a target  H_bar = ratio * log|A|
  * critic values are detached in the actor loss
  * explicit seeding
"""

from __future__ import annotations

import os

import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F


class ReplayBuffer:
    def __init__(self, obs_shape, capacity: int, rng: np.random.Generator):
        self.capacity, self.ptr, self.size, self.rng = capacity, 0, 0, rng
        self.obs = np.zeros((capacity, *obs_shape), np.float32)
        self.next_obs = np.zeros((capacity, *obs_shape), np.float32)
        self.act = np.zeros(capacity, np.int64)
        self.rew = np.zeros(capacity, np.float32)
        self.done = np.zeros(capacity, np.float32)

    def store(self, o, a, r, d, o2):
        i = self.ptr
        self.obs[i], self.act[i], self.rew[i], self.done[i], self.next_obs[i] = o, a, r, d, o2
        self.ptr, self.size = (i + 1) % self.capacity, min(self.size + 1, self.capacity)

    def sample(self, n):
        idx = self.rng.integers(0, self.size, n)
        return self.obs[idx], self.act[idx], self.rew[idx], self.done[idx], self.next_obs[idx]


class Encoder(nn.Module):
    """Conv(3->16,k4,s2) -> Conv(16->32,k2) -> FC128 -> FC64, ELU (as in the original code)."""

    def __init__(self, obs_shape):
        super().__init__()
        c, h, w = obs_shape
        self.conv = nn.Sequential(nn.Conv2d(c, 16, 4, 2), nn.ELU(), nn.Conv2d(16, 32, 2, 1), nn.ELU(), nn.Flatten())
        with T.no_grad():
            n = self.conv(T.zeros(1, c, h, w)).shape[1]
        self.fc = nn.Sequential(nn.Linear(n, 128), nn.ELU(), nn.Linear(128, 64), nn.ELU())

    def forward(self, x):
        return self.fc(self.conv(x))


class Head(nn.Module):
    def __init__(self, obs_shape, n_out):
        super().__init__()
        self.enc, self.out = Encoder(obs_shape), nn.Linear(64, n_out)

    def forward(self, x):
        return self.out(self.enc(x))


class DiscreteSAC:
    def __init__(self, obs_shape, n_actions, *, gamma=0.99, lr=3e-4, tau=0.005,
                 alpha=0.2, auto_alpha=True, target_entropy_ratio=0.5,
                 buffer_size=200_000, batch_size=128, seed=0, device=None):
        T.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.device = T.device(device or ("cuda" if T.cuda.is_available() else "cpu"))
        self.n_actions, self.gamma, self.tau, self.batch_size = n_actions, gamma, tau, batch_size
        self.buffer = ReplayBuffer(obs_shape, buffer_size, self.rng)

        mk = lambda: Head(obs_shape, n_actions).to(self.device)
        self.q1, self.q2, self.q1_t, self.q2_t, self.pi = mk(), mk(), mk(), mk(), mk()
        self.q1_t.load_state_dict(self.q1.state_dict())
        self.q2_t.load_state_dict(self.q2.state_dict())
        self.q_opt = T.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr)
        self.pi_opt = T.optim.Adam(self.pi.parameters(), lr=lr)

        self.auto_alpha = auto_alpha
        self.target_entropy = target_entropy_ratio * float(np.log(n_actions))
        self.log_alpha = T.tensor(np.log(alpha), dtype=T.float32, device=self.device, requires_grad=auto_alpha)
        self.alpha_opt = T.optim.Adam([self.log_alpha], lr=lr) if auto_alpha else None

    @property
    def alpha(self) -> float:
        return float(self.log_alpha.detach().exp())

    def _policy(self, x):
        logits = self.pi(x)
        return F.softmax(logits, -1), F.log_softmax(logits, -1)

    def act(self, obs, greedy=False) -> int:
        with T.no_grad():
            x = T.as_tensor(obs, device=self.device).unsqueeze(0)
            probs, _ = self._policy(x)
            if greedy:
                return int(probs.argmax(-1))
            return int(T.multinomial(probs, 1))

    def update(self):
        o, a, r, d, o2 = (T.as_tensor(v, device=self.device) for v in self.buffer.sample(self.batch_size))
        alpha = self.log_alpha.exp().detach()

        # soft Bellman target: y = r + gamma (1-d) sum_a' pi(a'|s') [min Q_t(s',a') - alpha log pi(a'|s')]
        with T.no_grad():
            p2, lp2 = self._policy(o2)
            q_next = T.min(self.q1_t(o2), self.q2_t(o2))
            v_next = (p2 * (q_next - alpha * lp2)).sum(-1)
            y = r + self.gamma * (1 - d) * v_next

        q1 = self.q1(o).gather(1, a[:, None]).squeeze(1)
        q2 = self.q2(o).gather(1, a[:, None]).squeeze(1)
        q_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        self.q_opt.zero_grad()
        q_loss.backward()
        self.q_opt.step()

        # policy: minimise E_s sum_a pi(a|s) [alpha log pi(a|s) - min Q(s,a)]
        p, lp = self._policy(o)
        with T.no_grad():
            q_min = T.min(self.q1(o), self.q2(o))
        pi_loss = (p * (alpha * lp - q_min)).sum(-1).mean()
        self.pi_opt.zero_grad()
        pi_loss.backward()
        self.pi_opt.step()

        entropy = -(p * lp).sum(-1).detach()
        if self.auto_alpha:
            # J(alpha) = E[alpha (H(pi) - H_bar)], gradient step on log alpha
            a_loss = (self.log_alpha.exp() * (entropy - self.target_entropy)).mean()
            self.alpha_opt.zero_grad()
            a_loss.backward()
            self.alpha_opt.step()

        with T.no_grad():
            for net, tgt in ((self.q1, self.q1_t), (self.q2, self.q2_t)):
                for w, wt in zip(net.parameters(), tgt.parameters()):
                    wt.mul_(1 - self.tau).add_(self.tau * w)

        return {"q_loss": q_loss.item(), "pi_loss": pi_loss.item(),
                "entropy": entropy.mean().item(), "alpha": self.alpha}

    def save(self, path):
        T.save({k: getattr(self, k).state_dict() for k in ("q1", "q2", "q1_t", "q2_t", "pi")}
               | {"log_alpha": self.log_alpha.detach().cpu()}, path)

    def save_checkpoint(self, path, extra: dict):
        """Full training state (networks, optimisers, RNGs, replay buffer), written atomically."""
        b = self.buffer
        state = {
            "nets": {k: getattr(self, k).state_dict() for k in ("q1", "q2", "q1_t", "q2_t", "pi")},
            "q_opt": self.q_opt.state_dict(), "pi_opt": self.pi_opt.state_dict(),
            "alpha_opt": self.alpha_opt.state_dict() if self.alpha_opt else None,
            "log_alpha": self.log_alpha.detach().cpu(),
            "np_rng": self.rng.bit_generator.state,
            "torch_rng": T.get_rng_state(),
            "cuda_rng": T.cuda.get_rng_state_all() if T.cuda.is_available() else None,
            "buffer": {"ptr": b.ptr, "size": b.size,
                       "obs": T.from_numpy(b.obs[:b.size]),
                       "next_obs": T.from_numpy(b.next_obs[:b.size]),
                       "act": T.from_numpy(b.act[:b.size]), "rew": T.from_numpy(b.rew[:b.size]),
                       "done": T.from_numpy(b.done[:b.size])},
            "extra": extra,
        }
        tmp = path + ".tmp"
        T.save(state, tmp)
        os.replace(tmp, path)

    def load_checkpoint(self, path) -> dict:
        s = T.load(path, map_location="cpu", weights_only=False)  # weights move to the device on load_state_dict
        for k, v in s["nets"].items():
            getattr(self, k).load_state_dict(v)
        self.q_opt.load_state_dict(s["q_opt"])
        self.pi_opt.load_state_dict(s["pi_opt"])
        if self.alpha_opt and s["alpha_opt"]:
            self.alpha_opt.load_state_dict(s["alpha_opt"])
        with T.no_grad():
            self.log_alpha.copy_(s["log_alpha"].to(self.device))
        self.rng.bit_generator.state = s["np_rng"]
        T.set_rng_state(s["torch_rng"].cpu())
        if s["cuda_rng"] is not None and T.cuda.is_available():
            T.cuda.set_rng_state_all([x.cpu() for x in s["cuda_rng"]])
        b, sb = self.buffer, s["buffer"]
        n = sb["size"]
        b.obs[:n] = sb["obs"].numpy()
        b.next_obs[:n] = sb["next_obs"].numpy()
        b.act[:n], b.rew[:n], b.done[:n] = sb["act"].numpy(), sb["rew"].numpy(), sb["done"].numpy()
        b.ptr, b.size = sb["ptr"], n
        return s["extra"]

    def load(self, path):
        ck = T.load(path, map_location=self.device)
        for k in ("q1", "q2", "q1_t", "q2_t", "pi"):
            getattr(self, k).load_state_dict(ck[k])
        with T.no_grad():
            self.log_alpha.copy_(ck["log_alpha"].to(self.device))
