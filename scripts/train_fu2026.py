"""Train the adapted QM-ATSC (base paper) Q-table at one penetration rate.

    python scripts/train_fu2026.py --p 0.2 --seed 0 --episodes 300 --out runs/fu2026/p0.2_s0
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.train import TRAIN_SEED_BASE  # noqa: E402
from tsc.env import EnvConfig  # noqa: E402
from tsc.make import NETWORKS, make_env  # noqa: E402
from tsc.fu2026 import QMATSC  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=float, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--net", choices=NETWORKS, default="1tls_3x3")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    json.dump(vars(args) | {"method": "fu2026_qm_atsc"}, open(os.path.join(args.out, "config.json"), "w"), indent=2)

    env = make_env(EnvConfig(p=args.p, reward="privileged", workdir=os.path.join(args.out, "sumo")), args.net)
    ctrl = QMATSC(env, seed=args.seed)
    t0 = time.time()
    with open(os.path.join(args.out, "train_log.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "true_cost_mean", "switch_rate", "epsilon", "wall_s"])
        for ep in range(args.episodes):
            obs, done, cost, sw, n = env.reset(seed=TRAIN_SEED_BASE * (args.seed + 1) + ep), False, 0.0, 0, 0
            while not done:
                obs, _, done, info = env.step(ctrl.act(obs))
                cost += info["true_cost"]
                sw += info["switched"]
                n += 1
            w.writerow([ep, round(cost / n, 3), round(sw / n, 3), round(ctrl.epsilon, 3), round(time.time() - t0, 1)])
            f.flush()
            ctrl.end_episode()
    ctrl.save(os.path.join(args.out, "qtable.npy"))
    env.close()
    open(os.path.join(args.out, "DONE"), "w").write(f"{args.episodes} episodes, {time.time() - t0:.0f} s\n")


if __name__ == "__main__":
    main()
