"""Train discrete SAC (default) or Double DQN on the partially detected intersection.

Example:
    python scripts/train.py --reward stratified --p 0.2 --seed 0 --out runs/sac_stratified_p0.2_s0
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tsc.env import EnvConfig  # noqa: E402
from tsc.make import NETWORKS, make_env  # noqa: E402
from tsc.pressure import FusedReward, PressureReward  # noqa: E402
from tsc.rewards import REWARD_MODES  # noqa: E402
from tsc.dqn import DoubleDQN  # noqa: E402
from tsc.sac import DiscreteSAC  # noqa: E402

TRAIN_SEED_BASE = 1_000_000  # episode seed = TRAIN_SEED_BASE * (run seed + 1) + episode index


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reward", choices=REWARD_MODES + ("pressure", "fused", "fused_cs"), required=True)
    ap.add_argument("--fused-lambda", type=float, default=1.0, help="weight of the pressure term (--reward fused)")
    ap.add_argument("--p", type=float, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=180_000, help="decision steps")
    ap.add_argument("--warmup", type=int, default=10_000, help="random-action steps before learning")
    ap.add_argument("--alpha", default="auto", help="'auto' or a fixed temperature, e.g. 0.05")
    ap.add_argument("--alpha-init", type=float, default=0.2)
    ap.add_argument("--target-entropy-ratio", type=float, default=0.5)
    ap.add_argument("--reward-scale", type=float, default=50.0)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lr", type=float, default=None, help="default 3e-4 (SAC) / 1e-4 (DQN)")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--device", default=None)
    ap.add_argument("--ckpt-every", type=int, default=50, help="episodes between resumable checkpoints")
    ap.add_argument("--algo", choices=["sac", "dqn"], default="sac")
    ap.add_argument("--net", choices=NETWORKS, default="1tls_3x3")
    ap.add_argument("--p-mult", default=None, help="per-approach multipliers of p, e.g. 0.5,1.5,0.75,1.25")
    ap.add_argument("--demand", choices=["random", "fixed", "peak"], default="random")
    ap.add_argument("--obs-loop", action="store_true", help="add loop-detector counts to the state")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.lr is None:
        args.lr = 3e-4 if args.algo == "sac" else 1e-4
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    p_mult = tuple(float(x) for x in args.p_mult.split(",")) if args.p_mult else None
    # pressure (PressLight) replaces the env reward; the env's own estimate is then unused.
    # fused / fused_cs = the env's stratified (collapsed-strata) cost plus lambda * pressure
    env_reward = {"pressure": "loop_count", "fused": "stratified", "fused_cs": "stratified_cs"}.get(args.reward, args.reward)
    env = make_env(EnvConfig(p=args.p, reward=env_reward, reward_scale=args.reward_scale, p_mult=p_mult,
                             demand=args.demand, obs_loop=args.obs_loop, workdir=os.path.join(args.out, "sumo")), args.net)
    if args.reward == "pressure":
        env = PressureReward(env, args.reward_scale)
    elif args.reward in ("fused", "fused_cs"):
        env = FusedReward(env, args.reward_scale, args.fused_lambda)
    if args.algo == "dqn":
        agent = DoubleDQN(env.obs_shape, env.n_actions, gamma=args.gamma, lr=args.lr,
                          batch_size=args.batch_size, seed=args.seed, device=args.device)
    else:
        auto = args.alpha == "auto"
        agent = DiscreteSAC(env.obs_shape, env.n_actions, gamma=args.gamma, lr=args.lr,
                            alpha=args.alpha_init if auto else float(args.alpha), auto_alpha=auto,
                            target_entropy_ratio=args.target_entropy_ratio,
                            batch_size=args.batch_size, seed=args.seed, device=args.device)

    header = ["episode", "step", "return", "true_cost_mean", "switch_rate",
              "alpha", "entropy", "q_loss", "wall_s"]
    log_path = os.path.join(args.out, "train_log.csv")
    ckpt_path = os.path.join(args.out, "checkpoint.pt")
    t0, step, ep, wall0 = time.time(), 0, 0, 0.0

    kept = []
    if os.path.exists(ckpt_path):
        # resume after an interruption (e.g. power cut): restore everything saved at an episode boundary
        extra = agent.load_checkpoint(ckpt_path)
        step, ep, wall0 = extra["step"], extra["episode"], extra["wall_s"]
        if os.path.exists(log_path):
            with open(log_path, "rb") as f:
                lines = f.read().replace(b"\x00", b"").decode().splitlines()[1:]
            kept = [ln for ln in lines if ln and ln.split(",")[0].isdigit() and int(ln.split(",")[0]) < ep]
        print(f"resumed from episode {ep}, step {step}", flush=True)
    log = open(log_path, "w", newline="")
    log.write(",".join(header) + "\n" + "".join(ln + "\n" for ln in kept))
    w = csv.writer(log)
    log.flush()

    while step < args.steps:
        obs = env.reset(seed=TRAIN_SEED_BASE * (args.seed + 1) + ep)
        ret, costs, switches, n, stats = 0.0, 0.0, 0, 0, {}
        done = False
        while not done:
            a = int(agent.rng.integers(env.n_actions)) if step < args.warmup else agent.act(obs)
            obs2, r, done, info = env.step(a)
            # the episode ends on a time limit, not a terminal state: do not cut the bootstrap
            agent.buffer.store(obs, a, r, 0.0, obs2)
            obs = obs2
            ret += r
            costs += info["true_cost"]
            switches += info["switched"]
            n += 1
            step += 1
            if step >= args.warmup:
                stats = agent.update()
        w.writerow([ep, step, round(ret, 4), round(costs / n, 3), round(switches / n, 3),
                    round(agent.alpha, 5), round(stats.get("entropy", float("nan")), 4),
                    round(stats.get("q_loss", float("nan")), 5), round(wall0 + time.time() - t0, 1)])
        log.flush()
        os.fsync(log.fileno())
        ep += 1
        if ep % args.ckpt_every == 0 and step < args.steps:
            agent.save_checkpoint(ckpt_path, {"step": step, "episode": ep, "wall_s": wall0 + time.time() - t0})

    agent.save(os.path.join(args.out, "model.pt"))
    env.close()
    with open(os.path.join(args.out, "DONE"), "w") as f:
        f.write(f"{step} steps, {ep} episodes, {wall0 + time.time() - t0:.0f} s\n")
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)  # the replay buffer is large; the final model is in model.pt


if __name__ == "__main__":
    main()
