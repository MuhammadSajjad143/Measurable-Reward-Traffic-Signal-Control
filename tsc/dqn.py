"""Double DQN, used to check Proposition 4 (greedy learners are scale-invariant).

Same CNN encoder and replay buffer as tsc/sac.py. Settings:
  Double DQN target, Huber loss, hard target update every 1,000 updates,
  epsilon decays linearly 1.0 -> 0.05 over the first 50k agent steps (train.py
  handles the random warm-up itself, so the count starts after it),
  lr 1e-4 (value of the original code), batch 128, gamma 0.99, buffer 200k.
"""

from __future__ import annotations

import os

import numpy as np
import torch as T
import torch.nn.functional as F

from .sac import Head, ReplayBuffer


class DoubleDQN:
    def __init__(self, obs_shape, n_actions, *, gamma=0.99, lr=1e-4, buffer_size=200_000, batch_size=128,
                 target_every=1_000, eps_start=1.0, eps_end=0.05, eps_steps=50_000,
                 seed=0, device=None):
        T.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.device = T.device(device or ("cuda" if T.cuda.is_available() else "cpu"))
        self.n_actions, self.gamma, self.batch_size = n_actions, gamma, batch_size
        self.target_every, self.eps = target_every, (eps_start, eps_end, eps_steps)
        self.buffer = ReplayBuffer(obs_shape, buffer_size, self.rng)
        self.q, self.q_t = Head(obs_shape, n_actions).to(self.device), Head(obs_shape, n_actions).to(self.device)
        self.q_t.load_state_dict(self.q.state_dict())
        self.opt = T.optim.Adam(self.q.parameters(), lr=lr)
        self.steps, self.updates = 0, 0

    @property
    def alpha(self) -> float:
        return float("nan")

    def epsilon(self) -> float:
        start, end, n = self.eps
        frac = min(1.0, self.steps / n)
        return start + frac * (end - start)

    def act(self, obs, greedy=False) -> int:
        if not greedy:
            self.steps += 1
            if self.rng.random() < self.epsilon():
                return int(self.rng.integers(self.n_actions))
        with T.no_grad():
            return int(self.q(T.as_tensor(obs, device=self.device).unsqueeze(0)).argmax(-1))

    def update(self):
        o, a, r, d, o2 = (T.as_tensor(v, device=self.device) for v in self.buffer.sample(self.batch_size))
        with T.no_grad():
            a2 = self.q(o2).argmax(-1, keepdim=True)            # action chosen by the online net
            y = r + self.gamma * (1 - d) * self.q_t(o2).gather(1, a2).squeeze(1)  # valued by the target net
        q = self.q(o).gather(1, a[:, None]).squeeze(1)
        loss = F.smooth_l1_loss(q, y)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.updates += 1
        if self.updates % self.target_every == 0:
            self.q_t.load_state_dict(self.q.state_dict())
        return {"q_loss": loss.item(), "entropy": float("nan"), "alpha": float("nan"), "epsilon": self.epsilon()}

    def save(self, path):
        T.save({"q": self.q.state_dict(), "q_t": self.q_t.state_dict()}, path)

    def load(self, path):
        ck = T.load(path, map_location=self.device)
        self.q.load_state_dict(ck["q"])
        self.q_t.load_state_dict(ck["q_t"])

    def save_checkpoint(self, path, extra: dict):
        b = self.buffer
        state = {"q": self.q.state_dict(), "q_t": self.q_t.state_dict(), "opt": self.opt.state_dict(),
                 "steps": self.steps, "updates": self.updates, "np_rng": self.rng.bit_generator.state,
                 "torch_rng": T.get_rng_state(),
                 "cuda_rng": T.cuda.get_rng_state_all() if T.cuda.is_available() else None,
                 "buffer": {"ptr": b.ptr, "size": b.size, "obs": T.from_numpy(b.obs[:b.size]),
                            "next_obs": T.from_numpy(b.next_obs[:b.size]), "act": T.from_numpy(b.act[:b.size]),
                            "rew": T.from_numpy(b.rew[:b.size]), "done": T.from_numpy(b.done[:b.size])},
                 "extra": extra}
        T.save(state, path + ".tmp")
        os.replace(path + ".tmp", path)

    def load_checkpoint(self, path) -> dict:
        s = T.load(path, map_location="cpu", weights_only=False)
        self.q.load_state_dict(s["q"])
        self.q_t.load_state_dict(s["q_t"])
        self.opt.load_state_dict(s["opt"])
        self.steps, self.updates = s["steps"], s["updates"]
        self.rng.bit_generator.state = s["np_rng"]
        T.set_rng_state(s["torch_rng"].cpu())
        if s["cuda_rng"] is not None and T.cuda.is_available():
            T.cuda.set_rng_state_all([x.cpu() for x in s["cuda_rng"]])
        b, sb = self.buffer, s["buffer"]
        n = sb["size"]
        b.obs[:n], b.next_obs[:n] = sb["obs"].numpy(), sb["next_obs"].numpy()
        b.act[:n], b.rew[:n], b.done[:n] = sb["act"].numpy(), sb["rew"].numpy(), sb["done"].numpy()
        b.ptr, b.size = sb["ptr"], n
        return s["extra"]
