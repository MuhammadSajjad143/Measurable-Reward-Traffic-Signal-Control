"""Pressure reward (PressLight, Wei et al., KDD 2019), from loop-detector counts.

r = -|sum over movements (N_in - N_out)| / K, with the same counts (all vehicles
within con_range m, upstream zone on real networks) as the max-pressure baseline.
Loops see every vehicle, so this reward is measurable at any penetration rate.

PressureReward wraps a single-intersection environment and replaces its reward;
junction_pressure() is also used by the multi-intersection environment.

FusedReward (proposed reward) keeps the lane-stratified cost and adds the
pressure as a discharge term: r = -(C_str + lambda * P) / K. The cost measures the
congestion near the stop line; the pressure term also rewards discharging the
intersection, so queues that grow beyond the approach zone are not free.
"""

from __future__ import annotations

from .baselines import MaxPressure


def junction_pressure(mp: MaxPressure) -> float:
    if mp._movements is None:
        mp._build()
    pairs = set().union(*mp._movements)
    return abs(sum(mp._count(i, True) - mp._count(o, False) for i, o in pairs))


class PressureReward:
    def __init__(self, env, scale: float):
        self.env, self.scale = env, scale
        self._mp = MaxPressure(env, "loop")

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, seed: int):
        return self.env.reset(seed)

    def step(self, action: int):
        obs, _, done, info = self.env.step(action)
        return obs, -junction_pressure(self._mp) / self.scale, done, info


class FusedReward:
    def __init__(self, env, scale: float, lam: float):
        assert env.cfg.reward in ("stratified", "stratified_cs"), "the fused reward builds on the stratified cost"
        self.env, self.scale, self.lam = env, scale, lam
        self._mp = MaxPressure(env, "loop")

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, seed: int):
        return self.env.reset(seed)

    def step(self, action: int):
        obs, r, done, info = self.env.step(action)
        return obs, r - self.lam * junction_pressure(self._mp) / self.scale, done, info
