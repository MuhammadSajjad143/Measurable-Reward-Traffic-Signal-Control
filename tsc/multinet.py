"""Real multi-intersection networks from the RESCO benchmark (Cologne, 3 signals; Ingolstadt, 7 signals).

Each signal is a Junction with the same measurement, state and reward as
RealNetEnv (approach zones up to ZONE m, derived yellow / all-red states,
CVs as cloned "<type>__con" vehicle types, lane speed limit as v_ref).
One agent per junction (independent learners).

Junctions decide asynchronously: a junction that keeps its phase is ready again
after t_green s, one that switches after t_yellow + t_red + t_green s. step()
takes the actions of the ready junctions, simulates second by second until at
least one junction is ready, and returns observations and rewards of the ready
ones. A junction's reward is the estimated cost of its own approaches at the
end of its decision interval, as in the single-intersection case (or its
pressure, for the PressLight-style baseline).

A Junction has the attributes the baselines use (tl_id, green_phases,
phase_states, segments, green_idx, n_actions, cfg, is_cv_type), so FixedTime
and MaxPressure run on it unchanged.
"""

from __future__ import annotations

import copy
import os
import tempfile
import xml.etree.ElementTree as ET

import numpy as np

try:
    import libsumo as traci
except ImportError:  # pragma: no cover
    import traci

import sumolib

from .baselines import MaxPressure
from .env import SUMO_BIN, EnvConfig
from .pressure import junction_pressure
from .realnet import ZONE, _is_green
from .rewards import REWARD_MODES, estimate_cost

NET_ROOT = os.path.join(os.path.dirname(__file__), "net")
MULTI_SCENARIOS = {"cologne3": (25200, 28800), "ingolstadt7": (57600, 61200), "cologne8": (25200, 28800)}
MIN_ROWS = 6


class Junction:
    def __init__(self, cfg: EnvConfig, net, tl, stop_lanes: set, claimed: set):
        self.cfg = cfg
        self.tl_id = tl.getID()
        conns = tl.getConnections()
        self.lanes = sorted({c[0].getID() for c in conns})
        self.lane_links = {}
        for in_lane, _, idx in conns:
            self.lane_links.setdefault(in_lane.getID(), []).append(idx)

        prog = list(tl.getPrograms().values())[0]
        greens = []
        for s in (p.state for p in prog.getPhases()):
            if "y" not in s.lower() and any(_is_green(ch) for ch in s) and s not in greens:
                greens.append(s)
        self.phase_states = greens
        self.green_phases = list(range(len(greens)))
        self.n_actions = len(greens)

        # approach zones; never cross into another junction's incoming lanes
        self.segments = {}
        for lane_id in self.lanes:
            seg, frontier = [(lane_id, 0.0)], [(net.getLane(lane_id), 0.0)]
            claimed.add(lane_id)
            while frontier:
                lane, off = frontier.pop()
                reach = off + lane.getLength()
                if reach >= ZONE:
                    continue
                for up in lane.getIncoming():
                    uid = up.getID()
                    if uid.startswith(":") or uid in claimed or uid in stop_lanes:
                        continue
                    claimed.add(uid)
                    seg.append((uid, reach))
                    frontier.append((up, reach))
            self.segments[lane_id] = seg
        self.v_ref = {lane_id: net.getLane(lane_id).getSpeed() for lane_id in self.lanes}
        self.n_cells = int(cfg.con_range // cfg.cell_length)
        # the encoder (k4 s2, then k2) needs at least 6 lane rows: smaller junctions get empty rows
        self.obs_shape = (4 if cfg.obs_loop else 3, max(len(self.lanes), MIN_ROWS), self.n_cells)
        self.green_idx, self.plan, self.remaining = 0, [], 0

    def is_cv_type(self, type_id: str) -> bool:
        return type_id.endswith("__con")

    # --------------------------------------------------------------- control
    def _set_state(self, state):
        traci.trafficlight.setRedYellowGreenState(self.tl_id, state)

    def start(self, action: int) -> bool:
        cfg = self.cfg
        switched = action != self.green_idx
        new = self.phase_states[action]
        if switched:
            old = self.phase_states[self.green_idx]
            yellow = "".join("y" if _is_green(o) and not _is_green(n) else o for o, n in zip(old, new))
            red = "".join(o if _is_green(o) and _is_green(n) else "r" for o, n in zip(old, new))
            self.plan = [(yellow, cfg.t_yellow), (red, cfg.t_red), (new, cfg.t_green)]
            self.green_idx = action
        else:
            self.plan = [(new, cfg.t_green)]
        self._next()
        return switched

    def _next(self):
        state, self.remaining = self.plan.pop(0)
        self._set_state(state)

    def tick(self):
        """Call after every simulation second."""
        self.remaining -= 1
        if self.remaining == 0 and self.plan:
            self._next()

    @property
    def ready(self) -> bool:
        return self.remaining <= 0 and not self.plan

    # ----------------------------------------------------------- measurement
    def snapshot(self):
        costs, detected, speeds, dists = [], [], [], []
        for lane_id in self.lanes:
            v_ref = self.v_ref[lane_id]
            vs, ds, cv = [], [], []
            for seg_lane, off in self.segments[lane_id]:
                length = traci.lane.getLength(seg_lane)
                for vid in traci.lane.getLastStepVehicleIDs(seg_lane):
                    dist = off + length - traci.vehicle.getLanePosition(vid)
                    if dist > ZONE:
                        continue
                    vs.append(traci.vehicle.getSpeed(vid))
                    ds.append(dist)
                    cv.append(self.is_cv_type(traci.vehicle.getTypeID(vid)))
            v = np.array(vs, float)
            costs.append(1.0 - np.clip(v / v_ref, 0.0, 1.0) ** 2)
            detected.append(np.array(cv, bool))
            speeds.append(v)
            dists.append(np.array(ds, float))
        return {"costs": costs, "detected": detected, "speeds": speeds, "dists": dists}

    def lane_green(self):
        state = self.phase_states[self.green_idx]
        return [any(_is_green(state[i]) for i in self.lane_links.get(lane_id, [])) for lane_id in self.lanes]

    def observe(self, snap=None):
        snap = snap or self.snapshot()
        cfg = self.cfg
        obs = np.zeros(self.obs_shape, dtype=np.float32)
        state = self.phase_states[self.green_idx]
        for l, lane_id in enumerate(self.lanes):
            d, v, dist = snap["detected"][l], snap["speeds"][l], snap["dists"][l]
            for vi, di in zip(v[d], dist[d]):
                if di <= cfg.con_range:
                    cell = min(int(di / cfg.cell_length), self.n_cells - 1)
                    obs[0, l, cell] = 1.0
                    obs[1, l, cell] = vi / self.v_ref[lane_id]
            if any(_is_green(state[i]) for i in self.lane_links.get(lane_id, [])):
                obs[2, l, :] = 1.0
            if cfg.obs_loop:
                obs[3, l, :] = len(v) / 40.0
        return obs


class MultiRealNetEnv:
    def __init__(self, cfg: EnvConfig, net_name: str, fused_lambda: float = 1.0):
        if cfg.reward not in REWARD_MODES + ("pressure", "fused", "fused_cs"):
            raise ValueError(cfg.reward)
        self.cfg, self.net_name, self.fused_lambda = cfg, net_name, fused_lambda
        self.begin, self.end = MULTI_SCENARIOS[net_name]
        d = os.path.join(NET_ROOT, net_name)
        self.net_file = os.path.join(d, f"{net_name}.net.xml")
        self.src_rou = os.path.join(d, f"{net_name}.rou.xml")
        self.workdir = cfg.workdir or tempfile.mkdtemp(prefix="tsc_multi_")
        os.makedirs(self.workdir, exist_ok=True)
        self.rou_file = os.path.join(self.workdir, "episode.rou.xml")
        self.trip_file = os.path.join(self.workdir, "tripinfo.xml")

        net = sumolib.net.readNet(self.net_file, withPrograms=True)
        tls = sorted(net.getTrafficLights(), key=lambda t: t.getID())
        stop = {c[0].getID() for t in tls for c in t.getConnections()}
        claimed = set()
        self.junctions = [Junction(cfg, net, t, stop, claimed) for t in tls]
        self.n_agents = len(self.junctions)
        self._mp = [MaxPressure(j, "loop") for j in self.junctions]  # for the pressure reward
        self._started = False

    def is_cv_type(self, type_id: str) -> bool:
        return type_id.endswith("__con")

    def _write_routes(self, rng):
        tree = ET.parse(self.src_rou)
        root = tree.getroot()
        vtypes = {vt.get("id"): vt for vt in root.findall("vType")}
        movers = [el for el in root if el.tag in ("vehicle", "trip", "flow")]
        connected = rng.random(len(movers)) < self.cfg.p
        needed = set()
        for el, con in zip(movers, connected):
            t = el.get("type", "DEFAULT_VEHTYPE")
            if con:
                el.set("type", t + "__con")
                needed.add(t)
        for t in sorted(needed):
            clone = copy.deepcopy(vtypes[t]) if t in vtypes else ET.Element("vType")
            clone.set("id", t + "__con")
            if t not in vtypes:
                clone.set("vClass", "passenger")
            root.insert(0, clone)
        tree.write(self.rou_file)

    def _sumo_args(self, sumo_seed):
        args = [SUMO_BIN, "-n", self.net_file, "-r", self.rou_file, "-b", str(self.begin), "-e", str(self.end + 600),
                "--no-step-log", "true", "--no-warnings", "true", "--duration-log.disable", "true",
                "--time-to-teleport", str(self.cfg.episode_len), "--waiting-time-memory", str(self.cfg.episode_len),
                "--seed", str(sumo_seed)]
        if self.cfg.tripinfo:
            args += ["--tripinfo-output", self.trip_file, "--tripinfo-output.write-unfinished", "true"]
        return args

    def reset(self, seed: int):
        """Start an episode; returns {junction index: observation} (all ready)."""
        rng = np.random.default_rng(seed)
        self._write_routes(rng)
        args = self._sumo_args(seed % (2 ** 31))
        if self._started:
            traci.load(args[1:])
        else:
            traci.start(args)
            self._started = True
        for j in self.junctions:
            j.green_idx = 0
            j.start(0)
        self._run_until_ready()
        return {k: j.observe() for k, j in enumerate(self.junctions)}

    def _run_until_ready(self):
        while True:
            traci.simulationStep()
            for j in self.junctions:
                j.tick()
            if any(j.ready for j in self.junctions) or traci.simulation.getTime() >= self.end:
                return

    def step(self, actions: dict):
        """actions: {junction index: action} for every ready junction."""
        switched = sum(self.junctions[k].start(a) for k, a in actions.items())
        self._run_until_ready()
        obs, rew, true_cost = {}, {}, 0.0
        for k, j in enumerate(self.junctions):
            if not j.ready:
                continue
            snap = j.snapshot()
            if self.cfg.reward == "pressure":
                rew[k] = -junction_pressure(self._mp[k]) / self.cfg.reward_scale
            else:
                mode = {"fused": "stratified", "fused_cs": "stratified_cs"}.get(self.cfg.reward, self.cfg.reward)
                lg = j.lane_green() if mode == "stratified_cs" else None
                rew[k] = -estimate_cost(mode, snap["costs"], snap["detected"], self.cfg.p,
                                        self.cfg.prior_cost, lg) / self.cfg.reward_scale
                if self.cfg.reward in ("fused", "fused_cs"):
                    rew[k] -= self.fused_lambda * junction_pressure(self._mp[k]) / self.cfg.reward_scale
            obs[k] = j.observe(snap)
            true_cost += float(sum(c.sum() for c in snap["costs"]))
        done = traci.simulation.getTime() >= self.end
        info = {"true_cost": true_cost, "switched": switched, "n_decided": len(actions),
                "time": traci.simulation.getTime()}
        return obs, rew, done, info

    def close(self):
        if self._started:
            traci.close()
            self._started = False

    def read_tripinfo(self):
        self.close()
        rows = []
        for _, el in ET.iterparse(self.trip_file):
            if el.tag == "tripinfo":
                rows.append({"type": "con" if self.is_cv_type(el.get("vType")) else "def",
                             "duration": float(el.get("duration")), "time_loss": float(el.get("timeLoss")),
                             "waiting": float(el.get("waitingTime")), "arrived": float(el.get("arrival")) >= 0})
                el.clear()
        return rows
