"""QM-ATSC of Fu, Lu, Ding & Li (J. Traffic Transp. Eng. 2026), adapted to this intersection.

Original (2 phases, green time adjusted in 1 s steps, Q-learning per phase):
  relative traffic flow   D_i = m_i t_g               if approach i is green      (Eq. 2)
                          D_i = m_i t_r lambda(t_r)   if approach i is red
  red-time penalty        lambda(t_r) = exp(m_i l t_r / (L_i n_i))                 (Eq. 4)
  state                   order of D_i over the 4 approaches (24 states) x green time
  reward                  sum_{green} D_i - sum_{red} D_i                           (Eq. 5)
  CAV time adjustment     d_i = max_j p_j / L_i over low-speed CAVs on approach i   (Eq. 7)
                          shorten the green when d_i lambda(t_r) > eps, eps = 1     (Eq. 8)
  gamma 0.9, epsilon 1 -> 0.01 in steps of 0.01, max green 70 s.

Adaptation to our 4-phase intersection and decision interface (decide every 10 s green):
  * fixed cyclic phase order; action = extend the current phase (10 s) or move to the next
  * state = order of D_i (4! = 24 on the synthetic intersection, 3! = 6 on Ingolstadt)
    x elapsed green in 10 s units (1..7, max green 70 s)
  * m_i = arrivals per second on approach i, counted by loop detectors (all vehicles),
    smoothed with an exponential moving average
  * an approach is green when any of its lanes has a green signal
  * Eq. 8 forces a switch when a red approach has d_i lambda(t_r) > 1 (needs CAVs; inactive at p = 0)
  * learning rate 0.1 (not reported in the paper); epsilon decays by 0.01 per episode
"""

from __future__ import annotations

from itertools import permutations

import numpy as np

try:
    import libsumo as traci
except ImportError:  # pragma: no cover
    import traci

VEH_LEN = 5.0       # average vehicle length l (m)
LOW_SPEED = 2.0     # m/s, a CAV slower than this marks congestion (Eq. 7)
MAX_GREEN_UNITS = 7  # 70 s


class QMATSC:
    name = "fu2026_qm_atsc"

    def __init__(self, env, seed=0, lr=0.1, gamma=0.9, ema=0.3):
        self.env, self.lr, self.gamma, self.ema = env, lr, gamma, ema
        self.rng = np.random.default_rng(seed)
        self.epsilon = 1.0
        self.approaches = sorted({ln.rsplit("_", 1)[0] for ln in env.lanes})
        self.lanes_of = {a: [ln for ln in env.lanes if ln.rsplit("_", 1)[0] == a] for a in self.approaches}
        # state = order of D_i over the approaches (4! = 24 states on the synthetic intersection)
        self.orders = {perm: k for k, perm in enumerate(permutations(range(len(self.approaches))))}
        self.q = np.zeros((len(self.orders), MAX_GREEN_UNITS + 1, 2))
        # real networks: an approach covers its incoming lanes plus the upstream lanes in the
        # 160 m zone, and L_i is the zone length (some incoming lanes are under 10 m)
        segs = getattr(env, "segments", None)
        self.parts = {a: [(ln, 0.0) for ln in self.lanes_of[a]] if segs is None
                      else [part for ln in self.lanes_of[a] for part in segs[ln]] for a in self.approaches}
        self.zone = None if segs is None else 160.0
        self.reset_episode()

    # ------------------------------------------------------------ measurement
    def reset_episode(self):
        self.seen = {a: set() for a in self.approaches}
        self.m = {a: 0.0 for a in self.approaches}
        self.red_since = {a: 0.0 for a in self.approaches}
        self.green_units = 1
        self.last_t = None
        self.prev = None  # (state, action, reward-at-decision) for the Q update

    def _green_approaches(self):
        state = self.env.phase_states[self.env.green_phases[self.env.green_idx]]
        links = self.env.lane_links
        return {a for a in self.approaches
                if any(state[i] in "Gg" for ln in self.lanes_of[a] for i in links.get(ln, []))}

    def _measure(self):
        t = traci.simulation.getTime()
        dt = 10.0 if self.last_t is None else max(1.0, t - self.last_t)
        self.last_t = t
        green = self._green_approaches()
        D, d = {}, {}
        for a in self.approaches:
            ids = set()
            for ln, _ in self.parts[a]:
                ids.update(traci.lane.getLastStepVehicleIDs(ln))
            arrivals = len(ids - self.seen[a])  # loop detector at the approach entrance
            self.seen[a] |= ids
            self.m[a] = (1 - self.ema) * self.m[a] + self.ema * arrivals / dt
            if a in green:
                self.red_since[a] = t
                D[a] = self.m[a] * 10.0 * self.green_units
                d[a] = 0.0
            else:
                tr = t - self.red_since[a]
                L = self.zone or traci.lane.getLength(self.lanes_of[a][0])
                n = len(self.lanes_of[a])
                lam = float(np.exp(min(50.0, self.m[a] * VEH_LEN * tr / (L * n))))
                D[a] = self.m[a] * tr * lam
                # Eq. 7: farthest low-speed CAV from the stop line, as a share of the approach length
                far = 0.0
                for ln, off in self.parts[a]:
                    length = traci.lane.getLength(ln)
                    for v in traci.lane.getLastStepVehicleIDs(ln):
                        if self.env.is_cv_type(traci.vehicle.getTypeID(v)) and traci.vehicle.getSpeed(v) < LOW_SPEED:
                            far = max(far, off + length - traci.vehicle.getLanePosition(v))
                d[a] = min(far, L) / L * lam
        return D, d, green

    def _state(self, D):
        order = tuple(sorted(range(len(self.approaches)), key=lambda k: -D[self.approaches[k]]))
        return self.orders[order], min(self.green_units, MAX_GREEN_UNITS)

    # ----------------------------------------------------------------- policy
    def act(self, obs, greedy=False):
        D, d, green = self._measure()
        s = self._state(D)
        r = sum(D[a] for a in green) - sum(D[a] for a in self.approaches if a not in green)  # Eq. 5
        if not greedy and self.prev is not None:
            ps, pa = self.prev
            target = r + self.gamma * self.q[s].max()
            self.q[ps][pa] += self.lr * (target - self.q[ps][pa])
        if self.green_units >= MAX_GREEN_UNITS:
            a = 1                                      # max green reached: switch
        elif any(d[x] > 1.0 for x in self.approaches if x not in green):
            a = 1                                      # Eq. 8: CAV-detected congestion on a red approach
        elif not greedy and self.rng.random() < self.epsilon:
            a = int(self.rng.integers(2))
        else:
            a = int(self.q[s].argmax())
        self.prev = (s, a)
        self.green_units = 1 if a == 1 else self.green_units + 1
        return (self.env.green_idx + 1) % self.env.n_actions if a == 1 else self.env.green_idx

    def end_episode(self):
        self.epsilon = max(0.01, self.epsilon - 0.01)
        self.reset_episode()

    def save(self, path):
        np.save(path, self.q)

    def load(self, path):
        self.q = np.load(path)
