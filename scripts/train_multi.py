"""Train independent discrete SAC (or Double DQN) agents, one per signal, on a multi-intersection network.

Same agent, hyperparameters, rewards and state as the single-intersection runs;
each agent learns from the decisions of its own junction. The step budget counts
the decisions of each agent (training stops when every agent reached --steps).

Example:
    python scripts/train_multi.py --net cologne3 --obs-loop --reward stratified --p 0.1 --seed 0 \
        --out runs/multi/cologne3/sac_auto_loopstate/stratified_p0.1_s0
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tsc.env import EnvConfig  # noqa: E402
from tsc.multinet import MULTI_SCENARIOS, MultiRealNetEnv  # noqa: E402
from tsc.rewards import REWARD_MODES  # noqa: E402
from tsc.dqn import DoubleDQN  # noqa: E402
from tsc.sac import DiscreteSAC  # noqa: E402

TRAIN_SEED_BASE = 1_000_000


def save_all(agents, ckpt_dir, extra):
    # write a complete new set, then swap it in, so a power cut never leaves a mixed set
    new, old = ckpt_dir + ".new", ckpt_dir + ".old"
    shutil.rmtree(new, ignore_errors=True)
    os.makedirs(new)
    for k, ag in enumerate(agents):
        ag.save_checkpoint(os.path.join(new, f"agent{k}.pt"), extra)
    if os.path.exists(ckpt_dir):
        shutil.rmtree(old, ignore_errors=True)
        os.replace(ckpt_dir, old)
    os.replace(new, ckpt_dir)
    shutil.rmtree(old, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", choices=sorted(MULTI_SCENARIOS), default="cologne3")
    ap.add_argument("--reward", choices=REWARD_MODES + ("pressure", "fused", "fused_cs"), required=True)
    ap.add_argument("--fused-lambda", type=float, default=1.0)
    ap.add_argument("--algo", choices=["sac", "dqn"], default="sac")
    ap.add_argument("--p", type=float, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=180_000, help="decisions per agent")
    ap.add_argument("--warmup", type=int, default=10_000)
    ap.add_argument("--alpha-init", type=float, default=0.2)
    ap.add_argument("--target-entropy-ratio", type=float, default=0.5)
    ap.add_argument("--reward-scale", type=float, default=50.0)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lr", type=float, default=None, help="default 3e-4 (SAC) / 1e-4 (DQN)")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--device", default=None)
    ap.add_argument("--ckpt-every", type=int, default=50)
    ap.add_argument("--obs-loop", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.lr is None:
        args.lr = 3e-4 if args.algo == "sac" else 1e-4
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(dict(vars(args), alpha="auto"), f, indent=2)

    env = MultiRealNetEnv(EnvConfig(p=args.p, reward=args.reward, reward_scale=args.reward_scale,
                                    obs_loop=args.obs_loop, workdir=os.path.join(args.out, "sumo")), args.net,
                          fused_lambda=args.fused_lambda)
    if args.algo == "dqn":
        agents = [DoubleDQN(j.obs_shape, j.n_actions, gamma=args.gamma, lr=args.lr, batch_size=args.batch_size,
                            seed=args.seed * 100 + k, device=args.device) for k, j in enumerate(env.junctions)]
    else:
        agents = [DiscreteSAC(j.obs_shape, j.n_actions, gamma=args.gamma, lr=args.lr, alpha=args.alpha_init,
                              auto_alpha=True, target_entropy_ratio=args.target_entropy_ratio,
                              batch_size=args.batch_size, seed=args.seed * 100 + k, device=args.device)
                  for k, j in enumerate(env.junctions)]

    header = ["episode", "step", "return", "true_cost_mean", "switch_rate",
              "alpha", "entropy", "q_loss", "wall_s"]
    log_path = os.path.join(args.out, "train_log.csv")
    ckpt_dir = os.path.join(args.out, "checkpoint")
    if not os.path.exists(ckpt_dir) and os.path.exists(ckpt_dir + ".old"):
        os.replace(ckpt_dir + ".old", ckpt_dir)  # interrupted during the swap
    t0, ep, wall0 = time.time(), 0, 0.0
    steps = [0] * len(agents)

    kept = []
    if os.path.exists(ckpt_dir):
        for k, ag in enumerate(agents):
            extra = ag.load_checkpoint(os.path.join(ckpt_dir, f"agent{k}.pt"))
        steps, ep, wall0 = list(extra["steps"]), extra["episode"], extra["wall_s"]
        if os.path.exists(log_path):
            with open(log_path, "rb") as f:
                lines = f.read().replace(b"\x00", b"").decode().splitlines()[1:]
            kept = [ln for ln in lines if ln and ln.split(",")[0].isdigit() and int(ln.split(",")[0]) < ep]
        print(f"resumed from episode {ep}, steps {steps}", flush=True)
    log = open(log_path, "w", newline="")
    log.write(",".join(header) + "\n" + "".join(ln + "\n" for ln in kept))
    w = csv.writer(log)
    log.flush()

    def choose(k, o):
        ag = agents[k]
        return int(ag.rng.integers(env.junctions[k].n_actions)) if steps[k] < args.warmup else ag.act(o)

    while min(steps) < args.steps:
        obs = env.reset(seed=TRAIN_SEED_BASE * (args.seed + 1) + ep)
        last = {k: (o, choose(k, o)) for k, o in obs.items()}  # latest decision of each junction
        actions = {k: a for k, (_, a) in last.items()}
        ret, costs, switches, n_dec, n_steps, stats = 0.0, 0.0, 0, 0, 0, []
        done = False
        while not done:
            obs2, rew, done, info = env.step(actions)
            costs += info["true_cost"]
            switches += info["switched"]
            n_dec += info["n_decided"]
            n_steps += 1
            actions = {}
            for k, o2 in obs2.items():
                o, a = last[k]
                agents[k].buffer.store(o, a, rew[k], 0.0, o2)
                ret += rew[k]
                steps[k] += 1
                if steps[k] >= args.warmup:
                    stats.append(agents[k].update())
                last[k] = (o2, choose(k, o2))
                actions[k] = last[k][1]
        mean = lambda key: float(np.mean([s[key] for s in stats if key in s])) if stats else float("nan")
        w.writerow([ep, min(steps), round(ret, 4), round(costs / max(n_steps, 1), 3),
                    round(switches / max(n_dec, 1), 3), round(float(np.mean([a.alpha for a in agents])), 5),
                    round(mean("entropy"), 4), round(mean("q_loss"), 5), round(wall0 + time.time() - t0, 1)])
        log.flush()
        os.fsync(log.fileno())
        ep += 1
        if ep % args.ckpt_every == 0 and min(steps) < args.steps:
            save_all(agents, ckpt_dir, {"steps": steps, "episode": ep, "wall_s": wall0 + time.time() - t0})

    for k, ag in enumerate(agents):
        ag.save(os.path.join(args.out, f"model{k}.pt"))
    env.close()
    with open(os.path.join(args.out, "DONE"), "w") as f:
        f.write(f"{steps} steps per agent, {ep} episodes, {wall0 + time.time() - t0:.0f} s\n")
    shutil.rmtree(ckpt_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
