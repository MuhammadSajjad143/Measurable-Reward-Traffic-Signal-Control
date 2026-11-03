"""Non-learning controllers, using the same action interface as the agents.

fixed_time      cycle through the 4 green phases (10 s green each; "Uniform" in the code of Ducrocq and Farhi)
max_pressure    choose the phase with the largest (incoming - outgoing) count
                within 160 m; counts from CVs only ("cv") or loops ("loop")
"""

from __future__ import annotations

try:
    import libsumo as traci
except ImportError:  # pragma: no cover
    import traci


class FixedTime:
    name = "fixed_time"

    def __init__(self, env):
        self.env = env

    def act(self, obs, greedy=True):
        return (self.env.green_idx + 1) % self.env.n_actions


class MaxPressure:
    def __init__(self, env, source: str):
        assert source in ("cv", "loop")
        self.env, self.source, self.name = env, source, f"max_pressure_{source}"
        self._movements = None

    def _build(self):
        # for each green phase: the (in_lane, out_lane) pairs it serves
        links = traci.trafficlight.getControlledLinks(self.env.tl_id)
        self._movements = []
        for g in self.env.green_phases:
            state = self.env.phase_states[g]
            pairs = {(lk[0], lk[1]) for i, l in enumerate(links) for lk in l[:1] if state[i] in "Gg"}
            self._movements.append(pairs)

    def _count(self, lane, incoming: bool):
        rng = self.env.cfg.con_range
        # real networks: an incoming approach also covers the upstream lanes feeding it
        parts = self.env.segments[lane] if incoming and hasattr(self.env, "segments") else [(lane, 0.0)]
        n = 0
        for part, off in parts:
            length = traci.lane.getLength(part)
            for v in traci.lane.getLastStepVehicleIDs(part):
                if self.source == "cv" and not self.env.is_cv_type(traci.vehicle.getTypeID(v)):
                    continue
                pos = traci.vehicle.getLanePosition(v)
                if (off + length - pos if incoming else pos) <= rng:
                    n += 1
        return n

    def act(self, obs, greedy=True):
        if self._movements is None:
            self._build()
        best, best_p = 0, None
        for k, pairs in enumerate(self._movements):
            p = sum(self._count(i, True) - self._count(o, False) for i, o in pairs)
            if best_p is None or p > best_p:
                best, best_p = k, p
        return best


def make_baseline(name, env):
    if name == "fixed_time":
        return FixedTime(env)
    if name == "max_pressure_cv":
        return MaxPressure(env, "cv")
    if name == "max_pressure_loop":
        return MaxPressure(env, "loop")
    raise ValueError(name)
