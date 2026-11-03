"""Robustness of the reward estimators when the assumptions are violated.

Uses the same SUMO snapshots as estimator_study.py (max-pressure with loops on
held-out scenarios) and redraws the CV mask. Three stress tests:

  A. misspecified p  (A1, p known)    ipw uses p_hat = k * p, k in {0.5, 0.75, 1.25, 1.5, 2}
  B. uneven CV share (A1, identical)  approach a has p_a = m_a * p, multipliers a permutation
                                      of {0.5, 0.75, 1.25, 1.5} (mean p); ipw uses the mean p
  C. loop count errors (A2)           each lane count is N_l * (1 + e), e ~ N(0, s^2), rounded,
                                      s in {0.05, 0.10, 0.20}

For each: relative bias, relative RMSE and phase agreement (as in estimator_study.py).

    python scripts/estimator_robustness.py --episodes 10 --reps 30
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.estimator_study import collect  # noqa: E402
from tsc.rewards import estimate_cost  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P_LEVELS = (0.1, 0.2, 0.4)


def phase_values(costs, det, served, est):
    return [est([costs[l] for l in lanes], [det[l] for l in lanes]) for lanes in served]


def evaluate(snaps, served, draw_mask, estimators, reps, rng):
    """estimators: name -> f(costs, det) -> total cost estimate."""
    acc = {n: {"e": [], "t": [], "agree": []} for n in estimators}
    for s in snaps:
        true = sum(c.sum() for c in s)
        best = int(np.argmax([sum(s[l].sum() for l in lanes) for lanes in served]))
        for _ in range(reps):
            det = draw_mask(s, rng)
            for n, f in estimators.items():
                acc[n]["e"].append(f(s, det))
                acc[n]["t"].append(true)
                acc[n]["agree"].append(int(np.argmax(phase_values(s, det, served, f))) == best)
    out = {}
    for n, a in acc.items():
        e, t = np.array(a["e"]), np.array(a["t"])
        out[n] = {"rel_bias": e.sum() / t.sum() - 1, "rel_rmse": np.sqrt(np.mean((e - t) ** 2)) / t.mean(),
                  "phase_agreement": float(np.mean(a["agree"]))}
    return out


def bernoulli(p_lane):
    return lambda s, rng: [rng.random(len(c)) < p for c, p in zip(s, p_lane)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--reps", type=int, default=30)
    args = ap.parse_args()
    rng = np.random.default_rng(1)
    snaps, served, lanes = collect(args.episodes, return_lanes=True)
    snaps = [s for s in snaps if sum(len(c) for c in s) > 0]
    approach = [ln.rsplit("_", 1)[0] for ln in lanes]
    approaches = sorted(set(approach))
    print(f"{len(snaps)} snapshots, approaches {approaches}")
    rows = []

    for p in P_LEVELS:
        same_p = [p] * len(lanes)
        # A. misspecified p for ipw; stratified and naive do not use p
        est = {"stratified": lambda c, d: estimate_cost("stratified", c, d, p)}
        for k in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0):
            est[f"ipw_phat={k}p"] = (lambda kk: lambda c, d: estimate_cost("ipw", c, d, kk * p))(k)
        for n, r in evaluate(snaps, served, bernoulli(same_p), est, args.reps, rng).items():
            rows.append({"test": "A_misspecified_p", "p": p, "estimator": n, **r})

        # B. uneven penetration across approaches (mean p)
        mult = dict(zip(approaches, (0.5, 1.5, 0.75, 1.25)))
        p_lane = [min(1.0, mult[a] * p) for a in approach]
        est = {m: (lambda mm: lambda c, d: estimate_cost(mm, c, d, p))(m) for m in ("naive", "ipw", "stratified", "loop_count")}
        for n, r in evaluate(snaps, served, bernoulli(p_lane), est, args.reps, rng).items():
            rows.append({"test": "B_uneven_penetration", "p": p, "estimator": n, **r})

        # C. noisy loop counts for stratified (loop_count shown with the same noise)
        for sd in (0.05, 0.10, 0.20):
            def noisy(mode, sd=sd):
                def f(c, d):
                    out = 0.0
                    for cl, dl in zip(c, d):
                        n_obs = max(0, int(round(len(cl) * (1 + rng.normal(0, sd))))) if len(cl) else 0
                        if mode == "loop_count":
                            out += n_obs
                        elif len(cl):
                            pooled = np.concatenate([x[y] for x, y in zip(c, d)])
                            mean = cl[dl].mean() if dl.any() else (pooled.mean() if pooled.size else 0.5)
                            out += n_obs * mean
                    return out
                return f
            est = {f"stratified_loop_sd={sd}": noisy("stratified"), f"loop_count_sd={sd}": noisy("loop_count")}
            for n, r in evaluate(snaps, served, bernoulli(same_p), est, args.reps, rng).items():
                rows.append({"test": "C_loop_noise", "p": p, "estimator": n, **r})
        print(f"p={p} done", flush=True)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    df.to_csv(os.path.join(ROOT, "results", "estimator_robustness.csv"), index=False)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
