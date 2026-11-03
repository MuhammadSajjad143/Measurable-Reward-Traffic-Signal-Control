"""Estimator study on real SUMO snapshots (no learning involved).

1. Collect snapshots (per-lane vehicle costs) while max-pressure (loops)
   controls the intersection on held-out scenarios.
2. For each p, redraw the CV mask R times per snapshot (d_i ~ Bernoulli(p))
   and measure, for every estimator:
     - relative bias   E[C_hat]/C - 1
     - relative RMSE
     - IPW variance against the derived formula (1-p)/p * sum c_i^2
     - phase agreement: how often the phase with the largest estimated cost
       on its served lanes equals the phase chosen with the true cost
       (a policy-free view of the selection effect)

    python scripts/estimator_study.py --episodes 10 --reps 50
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tsc.baselines import make_baseline  # noqa: E402
from tsc.env import EnvConfig, TSCEnv  # noqa: E402
from tsc.rewards import estimate_cost  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SNAP_SEED_BASE = 800_000_000
MODES = ("naive", "ipw", "stratified", "loop_count")
P_GRID = (0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0)


def collect(episodes, return_lanes=False):
    env = TSCEnv(EnvConfig(p=1.0, reward="privileged", workdir=os.path.join(ROOT, "runs", "estimator_study", "sumo")))
    ctrl = make_baseline("max_pressure_loop", env)
    snaps = []
    for k in range(episodes):
        obs, done = env.reset(seed=SNAP_SEED_BASE + k), False
        while not done:
            obs, _, done, _ = env.step(ctrl.act(obs))
            snaps.append([c.copy() for c in env.snapshot()["costs"]])
    # lanes served by each green phase
    served = []
    for g in env.green_phases:
        state = env.phase_states[g]
        served.append([l for l, lane in enumerate(env.lanes) if any(state[i].lower() == "g" for i in env.lane_links[lane])])
    lanes = list(env.lanes)
    env.close()
    return (snaps, served, lanes) if return_lanes else (snaps, served)


def phase_choice(lane_costs, lane_det, served, mode, p):
    """Phase whose served lanes carry the largest estimated cost."""
    vals = []
    for lanes in served:
        lc = [lane_costs[l] for l in lanes]
        if mode == "privileged":
            vals.append(sum(c.sum() for c in lc))
        else:
            vals.append(estimate_cost(mode, lc, [lane_det[l] for l in lanes], p))
    return int(np.argmax(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--reps", type=int, default=50)
    args = ap.parse_args()
    out = os.path.join(ROOT, "results")
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(0)

    snaps, served = collect(args.episodes)
    snaps = [s for s in snaps if sum(len(c) for c in s) > 0]
    print(f"{len(snaps)} snapshots")

    rows = []
    for p in P_GRID:
        acc = {m: {"est": [], "true": [], "agree": []} for m in MODES}
        ipw_var_emp, ipw_var_th = [], []
        for s in snaps:
            true = sum(c.sum() for c in s)
            best_true = phase_choice(s, None, served, "privileged", p)
            ipw_draws = []
            for _ in range(args.reps):
                det = [rng.random(len(c)) < p for c in s]
                for m in MODES:
                    e = estimate_cost(m, s, det, p)
                    acc[m]["est"].append(e)
                    acc[m]["true"].append(true)
                    acc[m]["agree"].append(phase_choice(s, det, served, m, p) == best_true)
                    if m == "ipw":
                        ipw_draws.append(e)
            ipw_var_emp.append(np.var(ipw_draws))
            ipw_var_th.append((1 - p) / p * sum((c ** 2).sum() for c in s))
        for m in MODES:
            est, true = np.array(acc[m]["est"]), np.array(acc[m]["true"])
            rows.append({"p": p, "estimator": m,
                         "rel_bias": est.sum() / true.sum() - 1,
                         "rel_rmse": np.sqrt(np.mean((est - true) ** 2)) / true.mean(),
                         "phase_agreement": np.mean(acc[m]["agree"]),
                         "ipw_var_ratio_emp_over_theory": (np.mean(ipw_var_emp) / np.mean(ipw_var_th)) if m == "ipw" and p < 1 else np.nan})
        print(f"p={p} done")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "estimator_study.csv"), index=False)
    print(df.round(3).to_string(index=False))

    fig, axs = plt.subplots(1, 3, figsize=(12, 3.6))
    for m in MODES:
        g = df[df.estimator == m]
        axs[0].plot(g.p, g.rel_bias, marker="o", label=m)
        axs[1].plot(g.p, g.rel_rmse, marker="o", label=m)
        axs[2].plot(g.p, g.phase_agreement, marker="o", label=m)
    for ax, t in zip(axs, ("Relative bias of total cost", "Relative RMSE", "Agreement with true-cost phase choice")):
        ax.set_title(t, fontsize=10)
        ax.set_xlabel("CV penetration rate p")
        ax.grid(alpha=0.3)
    axs[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_estimator_study.png"), dpi=200)


if __name__ == "__main__":
    main()
