"""Evaluate a trained agent or a baseline on fixed held-out scenarios.

Every method sees exactly the same episodes (demand + which vehicles are
connected), generated from EVAL_SEED_BASE + k. Agents act greedily.

Examples:
    python scripts/evaluate.py --run runs/sac_stratified_p0.2_s0 --p 0.2
    python scripts/evaluate.py --baseline max_pressure_loop --p 0.2 --out results/baselines
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# libsumo keeps state between the episodes of one process, which makes the Cologne
# networks irreproducible; socket TraCI runs every episode in a fresh SUMO process
# (bit-identical results across runs, same speed)
sys.modules["libsumo"] = None

from tsc.baselines import make_baseline  # noqa: E402
from tsc.env import EnvConfig  # noqa: E402
from tsc.make import NETWORKS, make_env  # noqa: E402
from tsc.dqn import DoubleDQN  # noqa: E402
from tsc.fu2026 import QMATSC  # noqa: E402
from tsc.sac import DiscreteSAC  # noqa: E402

EVAL_SEED_BASE = 900_000_000

FIELDS = ["method", "reward", "train_p", "eval_p", "seed", "episode",
          "delay_all", "delay_cv", "delay_noncv", "delay_gap", "waiting_all",
          "travel_time", "arrived", "unfinished", "true_cost_mean", "switch_rate", "n_decisions"]


def summarise(rows):
    def mean(sel, key):
        vals = [r[key] for r in rows if sel(r)]
        return float(np.mean(vals)) if vals else float("nan")
    cv = lambda r: r["type"] == "con"
    ncv = lambda r: r["type"] != "con"
    every = lambda r: True
    d_cv, d_ncv = mean(cv, "time_loss"), mean(ncv, "time_loss")
    return {
        "delay_all": mean(every, "time_loss"),
        "delay_cv": d_cv,
        "delay_noncv": d_ncv,
        "delay_gap": d_ncv - d_cv,
        "waiting_all": mean(every, "waiting"),
        "travel_time": mean(lambda r: r["arrived"], "duration"),
        "arrived": sum(r["arrived"] for r in rows),
        "unfinished": sum(not r["arrived"] for r in rows),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", help="training run directory (with model.pt, config.json)")
    ap.add_argument("--baseline", choices=["fixed_time", "max_pressure_cv", "max_pressure_loop", "fu2026"])
    ap.add_argument("--qtable", help="Q-table of a trained fu2026 controller (with config.json beside it)")
    ap.add_argument("--demand", choices=["random", "fixed", "peak"], default="random",
                    help="evaluation demand; random = the 20 held-out random scenarios")
    ap.add_argument("--p-mult", default=None, help="evaluate with uneven CV share per approach")
    ap.add_argument("--demand-scale", type=float, default=1.0, help="real networks: scale the calibrated demand (SUMO --scale)")
    ap.add_argument("--loop-noise", type=float, default=0.0, help="test only: relative s.d. of loop-count noise")
    ap.add_argument("--loop-fail", action="store_true", help="test only: one incoming lane's loop reports 0 each episode")
    ap.add_argument("--net", choices=NETWORKS, default=None, help="network for baselines (runs use their own)")
    ap.add_argument("--p", type=float, required=True, help="evaluation penetration rate")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--out", help="output directory (default: the run directory)")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    assert bool(args.run) != bool(args.baseline), "give exactly one of --run / --baseline"

    out = args.out or args.run or os.path.dirname(args.qtable)
    os.makedirs(out, exist_ok=True)
    scenario = "-".join(x for x in (f"demand_{args.demand}" if args.demand != "random" else "",
                                     "uneven" if args.p_mult else "",
                                     f"scale{args.demand_scale:g}" if args.demand_scale != 1.0 else "",
                                     f"noise{args.loop_noise:g}" if args.loop_noise else "",
                                     "fail1" if args.loop_fail else "") if x)
    tag = f"p{args.p}" + (f"_{scenario}" if scenario else "")
    name = args.baseline or "agent"
    workdir = os.path.join(out, f"sumo_eval_{name}_{tag}")
    cfg = json.load(open(os.path.join(args.run, "config.json"))) if args.run else {}
    if args.qtable:
        cfg = {"net": json.load(open(os.path.join(os.path.dirname(args.qtable), "config.json"))).get("net", "1tls_3x3")}
    net = cfg.get("net") or args.net or "1tls_3x3"
    p_mult = tuple(float(x) for x in args.p_mult.split(",")) if args.p_mult else None
    extra = ["--scale", str(args.demand_scale)] if args.demand_scale != 1.0 else []
    env = make_env(EnvConfig(p=args.p, reward="privileged", tripinfo=True, workdir=workdir, demand=args.demand,
                             p_mult=p_mult, obs_loop=cfg.get("obs_loop", False), extra_sumo_args=extra), net)
    # the 20 random held-out scenarios, or 20 seeds of the fixed / peak demand pattern
    seed_base = EVAL_SEED_BASE if args.demand == "random" else EVAL_SEED_BASE + 10_000

    if args.run:
        if cfg.get("algo", "sac") == "dqn":
            agent = DoubleDQN(env.obs_shape, env.n_actions, seed=0, device=args.device, buffer_size=1)
            method = "dqn"
        else:
            agent = DiscreteSAC(env.obs_shape, env.n_actions, seed=0, device=args.device, buffer_size=1)
            method = "sac_auto" if cfg["alpha"] == "auto" else f"sac_alpha{cfg['alpha']}"
        method += "_loopstate" if cfg.get("obs_loop") else ""
        # sensitivity runs get their own label so they never mix with the main study
        if cfg.get("algo", "sac") == "sac" and cfg.get("target_entropy_ratio", 0.5) != 0.5:
            method += f"_H{cfg['target_entropy_ratio']}"
        if cfg.get("reward_scale", 50.0) != 50.0:
            method += f"_K{cfg['reward_scale']:g}"
        method += "_trained_uneven" if cfg.get("p_mult") else ""
        agent.load(os.path.join(args.run, "model.pt"))
        meta = {"method": method, "reward": cfg["reward"], "train_p": cfg["p"], "seed": cfg["seed"]}
        policy = agent
    elif args.baseline == "fu2026":
        qcfg = json.load(open(os.path.join(os.path.dirname(args.qtable), "config.json")))
        policy = QMATSC(env, seed=0)
        policy.load(args.qtable)
        meta = {"method": "fu2026_qm_atsc", "reward": "-", "train_p": qcfg["p"], "seed": qcfg["seed"]}
    else:
        policy = make_baseline(args.baseline, env)
        meta = {"method": args.baseline, "reward": "-", "train_p": "-", "seed": "-"}

    # loop-detector imperfections at test time: they change what the controller sees
    # (the loop-count state channel; max-pressure's counts), never the true traffic
    noisy = args.loop_noise > 0 or args.loop_fail
    noise_rng, failed = np.random.default_rng(0), [None]

    def noisy_count(n, lane_idx):
        if failed[0] is not None and lane_idx == failed[0]:
            return 0
        return max(0, int(round(n * (1 + args.loop_noise * noise_rng.standard_normal())))) if args.loop_noise else n

    def perturb(obs):
        if not noisy or obs.shape[0] < 4:
            return obs
        obs = obs.copy()
        for l in range(obs.shape[1]):
            obs[3, l, :] = noisy_count(int(round(obs[3, l, 0] * 40)), l) / 40.0
        return obs

    if noisy and getattr(policy, "source", None) == "loop":
        raw_count = policy._count
        lanes = list(env.lanes)
        policy._count = lambda lane, incoming: (noisy_count(raw_count(lane, incoming), lanes.index(lane))
                                                if incoming else noisy_count(raw_count(lane, incoming), -1))

    path = os.path.join(out, f"eval_{name}_{tag}.csv")
    with open(path + ".tmp", "w", newline="") as f:
        extra_cols = (["scenario"] if scenario else []) + (["net"] if net != "1tls_3x3" else [])
        w = csv.DictWriter(f, fieldnames=FIELDS + extra_cols)
        w.writeheader()
        for k in range(args.episodes):
            obs = env.reset(seed=seed_base + k)
            noise_rng = np.random.default_rng(seed_base + k)
            failed[0] = int(noise_rng.integers(len(env.lanes))) if args.loop_fail else None
            obs = perturb(obs)
            if hasattr(policy, "reset_episode"):
                policy.reset_episode()
            done, costs, switches, n = False, 0.0, 0, 0
            while not done:
                obs, _, done, info = env.step(policy.act(obs, greedy=True))
                obs = perturb(obs)
                costs += info["true_cost"]
                switches += info["switched"]
                n += 1
            row = dict(meta, **({"scenario": scenario} if scenario else {}), **({"net": net} if net != "1tls_3x3" else {}),
                       eval_p=args.p, episode=k, true_cost_mean=costs / n,
                       switch_rate=switches / n, n_decisions=n, **summarise(env.read_tripinfo()))
            w.writerow({k2: (round(v, 4) if isinstance(v, float) else v) for k2, v in row.items()})
            f.flush()
    env.close()
    os.replace(path + ".tmp", path)  # only complete results count as done
    print("wrote", path)


if __name__ == "__main__":
    main()
