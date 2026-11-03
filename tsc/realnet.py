"""Real single intersections from the RESCO benchmark (Cologne, Ingolstadt).

The interface matches TSCEnv (reset / step / observe / snapshot, plus the attributes
the baselines use), so the agents, rewards, baselines and evaluation are unchanged.

Differences from the synthetic intersection:
  * demand is the real calibrated one-hour demand of the scenario (the same trips in
    every episode); episodes differ in which vehicles are CVs and in the SUMO seed
    (driver imperfection)
  * an "approach" is an incoming lane of the signal plus the upstream lanes feeding it,
    up to ZONE metres from the stop line (some incoming lanes are only 9-60 m long, so
    the queue sits upstream); reward, loop counts and the DTSE all use this zone
  * the green phases are the scenario's own; yellow (3 s) and all-red (2 s) states are
    derived from each pair of phases (links leaving green get yellow, then red)
  * a CV keeps its original vehicle type parameters: it gets a cloned type "<type>__con"
  * the cost reference speed is each lane's speed limit
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

from .env import SUMO_BIN, EnvConfig
from .rewards import REWARD_MODES, estimate_cost

NET_ROOT = os.path.join(os.path.dirname(__file__), "net")
SCENARIOS = {"cologne1": (25200, 28800), "ingolstadt1": (57600, 61200)}
ZONE = 160.0


def _is_green(ch):
    return ch in "Gg"


class RealNetEnv:
    def __init__(self, cfg: EnvConfig, net_name: str):
        if cfg.reward not in REWARD_MODES:
            raise ValueError(cfg.reward)
        self.cfg, self.net_name = cfg, net_name
        self.begin, self.end = SCENARIOS[net_name]
        d = os.path.join(NET_ROOT, net_name)
        self.net_file = os.path.join(d, f"{net_name}.net.xml")
        self.src_rou = os.path.join(d, f"{net_name}.rou.xml")
        self.workdir = cfg.workdir or tempfile.mkdtemp(prefix="tsc_real_")
        os.makedirs(self.workdir, exist_ok=True)
        self.rou_file = os.path.join(self.workdir, "episode.rou.xml")
        self.trip_file = os.path.join(self.workdir, "tripinfo.xml")

        net = sumolib.net.readNet(self.net_file, withPrograms=True)
        tl = net.getTrafficLights()[0]
        self.tl_id = tl.getID()
        conns = tl.getConnections()
        self.lanes = sorted({c[0].getID() for c in conns})
        self.lane_links = {}
        for in_lane, _, idx in conns:
            self.lane_links.setdefault(in_lane.getID(), []).append(idx)
        self.out_lanes = sorted({c[1].getID() for c in conns})

        prog = list(tl.getPrograms().values())[0]
        states = [p.state for p in prog.getPhases()]
        greens = []
        for s in states:
            if "y" not in s.lower() and any(_is_green(ch) for ch in s) and s not in greens:
                greens.append(s)
        self.phase_states = greens                 # one state string per action
        self.green_phases = list(range(len(greens)))
        self.n_actions = len(greens)

        # approach zones: incoming lane + upstream lanes within ZONE m of the stop line
        self.segments = {}  # incoming lane -> list of (lane id, distance from that lane's end to the stop line)
        claimed = set()
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
                    if uid.startswith(":") or uid in claimed or uid in self.lanes:
                        continue
                    claimed.add(uid)
                    seg.append((uid, reach))
                    frontier.append((up, reach))
            self.segments[lane_id] = seg
        self.v_ref = {lane_id: net.getLane(lane_id).getSpeed() for lane_id in self.lanes}

        self.n_cells = int(cfg.con_range // cfg.cell_length)
        self.obs_shape = (4 if cfg.obs_loop else 3, len(self.lanes), self.n_cells)
        self._started = False
        self._type_ids = None

    # ------------------------------------------------------------- episode set-up
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
        for t in sorted(needed):  # identical parameters, different id
            clone = copy.deepcopy(vtypes[t]) if t in vtypes else ET.Element("vType")
            clone.set("id", t + "__con")
            if t not in vtypes:
                clone.set("vClass", "passenger")
            root.insert(0, clone)
        self.n_vehicles, self.n_connected = len(movers), int(connected.sum())
        tree.write(self.rou_file)

    def _sumo_args(self, sumo_seed):
        args = [SUMO_BIN, "-n", self.net_file, "-r", self.rou_file, "-b", str(self.begin), "-e", str(self.end + 600),
                "--no-step-log", "true", "--no-warnings", "true", "--duration-log.disable", "true",
                "--time-to-teleport", str(self.cfg.episode_len), "--waiting-time-memory", str(self.cfg.episode_len),
                "--seed", str(sumo_seed)]
        if self.cfg.tripinfo:
            args += ["--tripinfo-output", self.trip_file, "--tripinfo-output.write-unfinished", "true"]
        return args + list(self.cfg.extra_sumo_args)  # e.g. --scale for heavier demand (evaluation)

    def reset(self, seed: int):
        rng = np.random.default_rng(seed)
        self._write_routes(rng)
        args = self._sumo_args(seed % (2 ** 31))
        if self._started:
            traci.load(args[1:])
        else:
            traci.start(args)
            self._started = True
        self.green_idx = 0
        self._set_state(self.phase_states[0])
        self._advance(self.cfg.t_green)
        return self.observe()

    # --------------------------------------------------------------- control
    def _set_state(self, state):
        traci.trafficlight.setRedYellowGreenState(self.tl_id, state)

    def _advance(self, seconds):
        for _ in range(seconds):
            traci.simulationStep()

    def step(self, action: int):
        cfg = self.cfg
        switched = action != self.green_idx
        if switched:
            old, new = self.phase_states[self.green_idx], self.phase_states[action]
            yellow = "".join("y" if _is_green(o) and not _is_green(n) else o for o, n in zip(old, new))
            red = "".join(o if _is_green(o) and _is_green(n) else "r" for o, n in zip(old, new))
            self._set_state(yellow)
            self._advance(cfg.t_yellow)
            self._set_state(red)
            self._advance(cfg.t_red)
            self.green_idx = action
            self._set_state(new)
        self._advance(cfg.t_green)
        snap = self.snapshot()
        lg = self.lane_green() if cfg.reward == "stratified_cs" else None
        r = -estimate_cost(cfg.reward, snap["costs"], snap["detected"], cfg.p, cfg.prior_cost, lg) / cfg.reward_scale
        info = {"true_cost": float(sum(c.sum() for c in snap["costs"])),
                "n_veh": int(sum(len(c) for c in snap["costs"])), "switched": switched,
                "time": traci.simulation.getTime()}
        done = traci.simulation.getTime() >= self.end
        return self.observe(snap), r, done, info

    def close(self):
        if self._started:
            traci.close()
            self._started = False

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
