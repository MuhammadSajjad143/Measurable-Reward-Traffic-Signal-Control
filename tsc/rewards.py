"""Reward estimators for traffic signal control under partial detection.

Every estimator targets the same quantity, the true congestion cost of the
intersection at a decision instant,

    C = sum_l sum_{i in V_l} c_i,        c_i = 1 - min(v_i / v_ref, 1)^2,

where V_l is the set of vehicles on incoming lane l. Each vehicle is a
connected vehicle (CV) with probability p, independently (d_i ~ Bernoulli(p)).
The estimators differ only in what they are allowed to see:

    privileged   all vehicles (simulation only, as in prior work)
    naive        CVs only:                sum_l sum_{i in D_l} c_i
    ipw          CVs, Horvitz-Thompson:   (1/p) sum_l sum_{i in D_l} c_i
    stratified   CVs + loop counts N_l:   sum_l N_l * mean_{i in D_l} c_i
    loop_count   loop counts only:        sum_l N_l
    stratified_cs  as stratified, but a lane without a CV borrows the mean CV cost of
                 the lanes with the same signal indication (red or green) instead
                 of all lanes (collapsed strata, Cochran 1977)

The training reward is r = -C_hat / scale.
"""

from __future__ import annotations

import numpy as np

REWARD_MODES = ("privileged", "naive", "ipw", "stratified", "loop_count", "stratified_cs")


def vehicle_cost(speeds: np.ndarray, v_ref: float) -> np.ndarray:
    """Squared-delay cost in [0, 1]; 0 at free flow, 1 when stopped."""
    ratio = np.clip(np.asarray(speeds, dtype=float) / v_ref, 0.0, 1.0)
    return 1.0 - ratio ** 2


def estimate_cost(
    mode: str,
    lane_costs: list[np.ndarray],
    lane_detected: list[np.ndarray],
    p: float,
    prior_cost: float = 0.5,
    lane_green: list[bool] | None = None,
) -> float:
    """Estimate the total cost C from one snapshot.

    lane_costs[l]    costs c_i of all vehicles on lane l (length N_l)
    lane_detected[l] boolean mask of the same length, True for CVs
    p                penetration rate, needed only by ``ipw``
    prior_cost       fallback per-vehicle cost for ``stratified`` when no CV
                     is visible anywhere in the snapshot
    lane_green[l]    whether lane l currently has green (``stratified_cs`` only)
    """
    if mode == "privileged":
        return float(sum(c.sum() for c in lane_costs))

    if mode == "naive":
        return float(sum(c[d].sum() for c, d in zip(lane_costs, lane_detected)))

    if mode == "ipw":
        if p <= 0:
            raise ValueError("ipw needs p > 0")
        return float(sum(c[d].sum() for c, d in zip(lane_costs, lane_detected))) / p

    if mode == "loop_count":
        return float(sum(len(c) for c in lane_costs))

    if mode == "stratified":
        detected = [c[d] for c, d in zip(lane_costs, lane_detected)]
        pooled = np.concatenate(detected) if detected else np.empty(0)
        fallback = float(pooled.mean()) if pooled.size else prior_cost
        total = 0.0
        for c, cd in zip(lane_costs, detected):
            n_lane = len(c)  # loop-detector count, sees every vehicle
            if n_lane == 0:
                continue
            total += n_lane * (float(cd.mean()) if cd.size else fallback)
        return total

    if mode == "stratified_cs":
        if lane_green is None:
            raise ValueError("stratified_cs needs lane_green")
        detected = [c[d] for c, d in zip(lane_costs, lane_detected)]
        pooled_all = np.concatenate(detected) if detected else np.empty(0)
        fallback_all = float(pooled_all.mean()) if pooled_all.size else prior_cost
        fallback = {}
        for g in (True, False):
            same = [cd for cd, lg in zip(detected, lane_green) if lg == g]
            pooled = np.concatenate(same) if same else np.empty(0)
            fallback[g] = float(pooled.mean()) if pooled.size else fallback_all
        total = 0.0
        for c, cd, g in zip(lane_costs, detected, lane_green):
            if len(c) == 0:
                continue
            total += len(c) * (float(cd.mean()) if cd.size else fallback[bool(g)])
        return total

    raise ValueError(f"unknown reward mode {mode!r}; choose from {REWARD_MODES}")
