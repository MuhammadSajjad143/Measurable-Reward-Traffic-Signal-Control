"""Evaluate independent SAC agents or per-junction baselines on a multi-intersection network.

Same held-out episodes (EVAL_SEED_BASE + k), metrics and CSV columns as
evaluate.py; delays are network-wide (every trip in the scenario).

Examples:
    python scripts/evaluate_multi.py --run runs/multi/cologne3/sac_auto_loopstate/stratified_p0.1_s0 --p 0.1
    python scripts/evaluate_multi.py --baseline max_pressure_loop --net cologne3 --p 0.1 --out runs/multi/cologne3/baselines
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# socket TraCI, as in evaluate.py: a fresh SUMO process per episode keeps evaluation reproducible
sys.modules["libsumo"] = None

from scripts.evaluate import EVAL_SEED_BASE, FIELDS, summarise  # noqa: E402
from tsc.baselines import make_baseline  # noqa: E402
from tsc.env import EnvConfig  # noqa: E402
from tsc.multinet import MULTI_SCENARIOS, MultiRealNetEnv  # noqa: E402
from tsc.dqn import DoubleDQN  # noqa: E402
from tsc.sac import DiscreteSAC  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run")
    ap.add_argument("--baseline", choices=["fixed_time", "max_pressure_cv", "max_pressure_loop"])
    ap.add_argument("--net", choices=sorted(MULTI_SCENARIOS), default=None)
    ap.add_argument("--p", type=float, required=True)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--out")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    assert bool(args.run) != bool(args.baseline), "give exactly one of --run / --baseline"

    out = args.out or args.run
    os.makedirs(out, exist_ok=True)
    name = args.baseline or "agent"
    cfg = json.load(open(os.path.join(args.run, "config.json"))) if args.run else {}
    net = cfg.get("net") or args.net or "cologne3"
    env = MultiRealNetEnv(EnvConfig(p=args.p, reward="privileged", tripinfo=True,
                                    workdir=os.path.join(out, f"sumo_eval_{name}_p{args.p}"),
                                    obs_loop=cfg.get("obs_loop", False)), net)
    if args.run:
        policies = []
        dqn = cfg.get("algo", "sac") == "dqn"
        for k, j in enumerate(env.junctions):
            cls = DoubleDQN if dqn else DiscreteSAC
            ag = cls(j.obs_shape, j.n_actions, seed=0, device=args.device, buffer_size=1)
            ag.load(os.path.join(args.run, f"model{k}.pt"))
            policies.append(ag)
        method = ("idqn" if dqn else "isac_auto") + ("_loopstate" if cfg.get("obs_loop") else "")
        meta = {"method": method, "reward": cfg["reward"], "train_p": cfg["p"], "seed": cfg["seed"]}
    else:
        policies = [make_baseline(args.baseline, j) for j in env.junctions]
        meta = {"method": args.baseline, "reward": "-", "train_p": "-", "seed": "-"}

    path = os.path.join(out, f"eval_{name}_p{args.p}.csv")
    with open(path + ".tmp", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS + ["net"])
        w.writeheader()
        for k in range(args.episodes):
            obs = env.reset(seed=EVAL_SEED_BASE + k)
            done, costs, switches, n, n_dec = False, 0.0, 0, 0, 0
            while not done:
                obs, _, done, info = env.step({i: policies[i].act(o, greedy=True) for i, o in obs.items()})
                costs += info["true_cost"]
                switches += info["switched"]
                n_dec += info["n_decided"]
                n += 1
            row = dict(meta, net=net, eval_p=args.p, episode=k, true_cost_mean=costs / n,
                       switch_rate=switches / max(n_dec, 1), n_decisions=n_dec, **summarise(env.read_tripinfo()))
            w.writerow({k2: (round(v, 4) if isinstance(v, float) else v) for k2, v in row.items()})
            f.flush()
    env.close()
    os.replace(path + ".tmp", path)
    print("wrote", path)


if __name__ == "__main__":
    main()
