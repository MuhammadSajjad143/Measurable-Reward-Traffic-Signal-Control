"""Monte Carlo checks of the estimator results derived in METHODOLOGY.md.

Run:  .venv/bin/python -m pytest tests -q
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tsc.rewards import estimate_cost, vehicle_cost  # noqa: E402

RNG = np.random.default_rng(0)
LANES = [RNG.uniform(0, 1, n) for n in (12, 5, 30, 0, 8, 20)]  # fixed costs c_i
C_TRUE = sum(c.sum() for c in LANES)
SUM_SQ = sum((c ** 2).sum() for c in LANES)
M = 200_000  # Monte Carlo draws


def draws(p, mode, condition_all_lanes=False):
    out = []
    for _ in range(M):
        det = [RNG.random(len(c)) < p for c in LANES]
        if condition_all_lanes and any(len(c) and not d.any() for c, d in zip(LANES, det)):
            continue
        out.append(estimate_cost(mode, LANES, det, p))
    return np.array(out)


def test_cost_range():
    c = vehicle_cost(np.array([0.0, 7.0, 13.89, 20.0]), 13.89)
    assert c[0] == 1.0 and c[2] == 0.0 and c[3] == 0.0 and 0 < c[1] < 1


def test_all_estimators_exact_at_full_detection():
    det = [np.ones(len(c), bool) for c in LANES]
    for mode in ("privileged", "naive", "ipw", "stratified"):
        assert np.isclose(estimate_cost(mode, LANES, det, 1.0), C_TRUE)


def test_naive_is_scaled_by_p():
    p = 0.3
    x = draws(p, "naive")
    se = x.std() / np.sqrt(len(x))
    assert abs(x.mean() - p * C_TRUE) < 4 * se


def test_ipw_unbiased_and_variance_formula():
    p = 0.3
    x = draws(p, "ipw")
    se = x.std() / np.sqrt(len(x))
    assert abs(x.mean() - C_TRUE) < 4 * se
    var_theory = (1 - p) / p * SUM_SQ
    assert abs(x.var() / var_theory - 1) < 0.02


def test_stratified_unbiased_given_every_lane_observed():
    p = 0.4
    x = draws(p, "stratified", condition_all_lanes=True)
    se = x.std() / np.sqrt(len(x))
    assert abs(x.mean() - C_TRUE) < 4 * se


def test_collapsed_strata_equals_stratified_when_every_lane_has_a_cv():
    lanes = [c for c in LANES if len(c)]
    det = [np.zeros(len(c), bool) for c in lanes]
    for d in det:
        d[0] = True
    green = [i % 2 == 0 for i in range(len(lanes))]
    assert np.isclose(estimate_cost("stratified_cs", lanes, det, 0.2, lane_green=green),
                      estimate_cost("stratified", lanes, det, 0.2))


def test_collapsed_strata_borrows_from_lanes_with_the_same_signal():
    # red lane with a stopped CV, green lane with a moving CV, red lane without CV
    lanes = [np.array([1.0, 1.0]), np.array([0.0, 0.0]), np.array([1.0, 1.0, 1.0])]
    det = [np.array([True, False]), np.array([True, False]), np.zeros(3, bool)]
    green = [False, True, False]
    # the CV-less red lane borrows the red mean (1.0), not the pooled mean (0.5)
    assert np.isclose(estimate_cost("stratified_cs", lanes, det, 0.2, lane_green=green), 2 * 1.0 + 2 * 0.0 + 3 * 1.0)
    assert np.isclose(estimate_cost("stratified", lanes, det, 0.2), 2 * 1.0 + 2 * 0.0 + 3 * 0.5)
