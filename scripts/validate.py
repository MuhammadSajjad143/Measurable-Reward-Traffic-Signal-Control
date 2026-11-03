"""Measurable validation score of a trained single-intersection agent, for seed selection.

Runs the greedy policy on validation episodes (seeds disjoint from training and from the
held-out test scenarios) and records the mean *estimated* cost per decision under a
measurable estimator (default: lane-stratified), i.e. what an operator could compute on site
from CVs and loop counts. The true cost is written too, for diagnosis only. With --delay the
simulated delay of every trip is recorded as well; this is used to choose the weight of the
fused reward on the synthetic intersection only (never on the test scenarios).

    python scripts/validate.py --run runs/real/cologne1/sac_auto_loopstate/stratified_p0.1_s0
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.evaluate import summarise  # noqa: E402
from tsc.dqn import DoubleDQN  # noqa: E402
from tsc.env import EnvConfig  # noqa: E402
from tsc.make import make_env  # noqa: E402
from tsc.sac import DiscreteSAC  # noqa: E402

VAL_SEED_BASE = 800_000_000  # training uses 1e6 * (seed + 1) + episode, testing 9e8 + k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--estimator", default="stratified", choices=["stratified", "loop_count", "ipw", "naive"])
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--delay", action="store_true", help="also record the simulated trip delay")
    args = ap.parse_args()

    cfg = json.load(open(os.path.join(args.run, "config.json")))
    p = cfg["p"]
    path = os.path.join(args.run, f"val_{args.estimator}_p{p}" + ("_delay" if args.delay else "") + ".csv")
    if os.path.exists(path):
        return
    env = make_env(EnvConfig(p=p, reward=args.estimator, obs_loop=cfg.get("obs_loop", False), tripinfo=args.delay,
                             workdir=os.path.join(args.run, "sumo_val")), cfg.get("net") or "1tls_3x3")
    cls = DoubleDQN if cfg.get("algo", "sac") == "dqn" else DiscreteSAC
    agent = cls(env.obs_shape, env.n_actions, seed=0, device="cpu", buffer_size=1)
    agent.load(os.path.join(args.run, "model.pt"))
    with open(path + ".tmp", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "est_cost_mean", "true_cost_mean", "n_decisions"] + (["delay_all"] if args.delay else []))
        for k in range(args.episodes):
            obs, done, est, true, n = env.reset(seed=VAL_SEED_BASE + k), False, 0.0, 0.0, 0
            while not done:
                obs, r, done, info = env.step(agent.act(obs, greedy=True))
                est += -r * env.cfg.reward_scale
                true += info["true_cost"]
                n += 1
            extra = [round(summarise(env.read_tripinfo())["delay_all"], 4)] if args.delay else []
            w.writerow([k, round(est / n, 4), round(true / n, 4), n] + extra)
    env.close()
    os.replace(path + ".tmp", path)


if __name__ == "__main__":
    main()
