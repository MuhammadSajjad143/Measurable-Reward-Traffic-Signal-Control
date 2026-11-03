"""Isolated-intersection SUMO environment with partial vehicle detection.

Faithful to the environment of Ducrocq and Farhi (2023):
  * network 1tls_3x3 (4 approaches x 3 lanes, 4 green phases)
  * action = choose one of the 4 green phases; keeping the current phase
    gives 10 s of green, switching gives 3 s yellow + 2 s red + 10 s green
  * DTSE state (3 x 12 x 20): CV occupancy, CV speed, lane-is-green, built
    from CVs within 160 m of the stop line only
  * per-episode demand: each approach draws 100..1000 veh/h, Poisson
    headways, random turning shares

Changes for this study (see FIXES.md):
  * the penetration rate p is a fixed run parameter and each vehicle is a CV
    with probability p (Bernoulli), instead of p ~ U[0, 1] per episode
  * every episode is generated from an explicit seed, so evaluation scenarios
    are identical across methods
  * the reward is one of the estimators in rewards.py, with a fixed scale
    (no running-max normalisation)
  * phase timing is controlled explicitly (SUMO never auto-advances phases)
"""

from __future__ import annotations

import os
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from itertools import permutations

import numpy as np

try:  # libsumo is much faster; one simulation per process
    import libsumo as traci
except ImportError:  # pragma: no cover
    import traci

import sumo
import sumolib

from .rewards import REWARD_MODES, estimate_cost, vehicle_cost

NET_DIR = os.path.join(os.path.dirname(__file__), "net", "1tls_3x3")
SUMO_BIN = os.path.join(sumo.SUMO_HOME, "bin", "sumo")


@dataclass
class EnvConfig:
    p: float = 0.2                 # CV penetration rate
    reward: str = "stratified"     # one of REWARD_MODES
    reward_scale: float = 50.0     # r = -C_hat / reward_scale
    episode_len: int = 3600        # simulated seconds
    t_green: int = 10
    t_yellow: int = 3
    t_red: int = 2
    con_range: float = 160.0       # CV state range (m)
    cell_length: float = 8.0
    v_ref: float = 13.89           # lane speed limit (m/s), cost is 0 at free flow
    v_max_vehicle: float = 16.67   # vehicle type maxSpeed (original value)
    flow_choices: tuple = tuple(range(100, 1001, 100))
    prior_cost: float = 0.5
    # --- extensions (defaults reproduce the main study exactly) ---
    p_mult: tuple | None = None    # per-approach multipliers of p, in sorted incoming-edge order
    demand: str = "random"         # "random" (per-episode draw), "fixed", or "peak"
    demand_flows: tuple = (200, 800, 800, 200)  # veh/h per approach for "fixed"/"peak"
    obs_loop: bool = False         # add a 4th state channel with loop-detector counts
    workdir: str | None = None
    tripinfo: bool = False         # write per-vehicle trip info (evaluation)
    extra_sumo_args: list = field(default_factory=list)


class TSCEnv:
    def __init__(self, cfg: EnvConfig):
        if cfg.reward not in REWARD_MODES:
            raise ValueError(cfg.reward)
        self.cfg = cfg
        self.workdir = cfg.workdir or tempfile.mkdtemp(prefix="tsc_")
        os.makedirs(self.workdir, exist_ok=True)
        self.net_file = os.path.join(NET_DIR, "1tls_3x3.net.xml")
        self.tll_file = os.path.join(NET_DIR, "1tls_3x3.tll.xml")
        self.rou_file = os.path.join(self.workdir, "episode.rou.xml")
        self.trip_file = os.path.join(self.workdir, "tripinfo.xml")

        net = sumolib.net.readNet(self.net_file)
        self.tl_id = net.getTrafficLights()[0].getID()
        node = net.getNode(self.tl_id)
        self.in_edges = sorted(e.getID() for e in node.getIncoming())
        self.lanes = [l.getID() for e in self.in_edges for l in net.getEdge(e).getLanes()]
        self.out_lanes = sorted(l.getID() for e in node.getOutgoing() for l in e.getLanes())

        # routes: every boundary node to every other boundary node
        boundary = [n.getID() for n in net.getNodes() if n.getID() != self.tl_id]
        self.routes = {}
        for o, d in permutations(boundary, 2):
            first = net.getNode(o).getOutgoing()[0]
            last = net.getNode(d).getIncoming()[0]
            edges = [e.getID() for e in net.getShortestPath(first, last)[0]]
            self.routes[f"{o}_{d}"] = edges
        # turning weights: number of lane links from each origin edge to each exit edge,
        # which reproduces the original split (uniform spacings -> Dirichlet(link counts))
        self.origin_routes = {}
        for rid, edges in self.routes.items():
            self.origin_routes.setdefault(edges[0], []).append(rid)
        self.link_weight = {}
        for (in_lane, out_lane, _) in [c for c in net.getTrafficLights()[0].getConnections()]:
            key = (in_lane.getEdge().getID(), out_lane.getEdge().getID())
            self.link_weight[key] = self.link_weight.get(key, 0) + 1

        self._started = False
        self.n_cells = int(cfg.con_range // cfg.cell_length)
        self.obs_shape = (4 if cfg.obs_loop else 3, len(self.lanes), self.n_cells)
        self._read_phases()
        self.n_actions = len(self.green_phases)

    # ------------------------------------------------------------------ setup
    def _read_phases(self):
        root = ET.parse(self.tll_file).getroot()
        states = [ph.get("state") for ph in root.iter("phase")]
        self.phase_states = states
        self.green_phases = [i for i, s in enumerate(states) if "g" in s.lower()]
        # lane -> list of link indices it controls (filled after the first start)
        self.lane_links = None

    def _sumo_args(self, sumo_seed: int):
        args = [
            SUMO_BIN, "-n", self.net_file, "-r", self.rou_file, "-a", self.tll_file,
            "--no-step-log", "true", "--no-warnings", "true", "--duration-log.disable", "true",
            "--time-to-teleport", str(self.cfg.episode_len),
            "--waiting-time-memory", str(self.cfg.episode_len),
            "--seed", str(sumo_seed),
        ]
        if self.cfg.tripinfo:
            args += ["--tripinfo-output", self.trip_file, "--tripinfo-output.write-unfinished", "true"]
        return args + list(self.cfg.extra_sumo_args)

    def _write_routes(self, rng: np.random.Generator):
        cfg = self.cfg
        vehicles = []
        self.flows = {}
        for k, e in enumerate(sorted(self.origin_routes)):
            if cfg.demand == "random":
                q = int(rng.choice(cfg.flow_choices))  # veh/h on this approach
            else:
                q = int(cfg.demand_flows[k])
            self.flows[e] = q
            rids = sorted(self.origin_routes[e])
            w = np.array([self.link_weight.get((e, self.routes[r][-1]), 1) for r in rids], float)
            shares = rng.dirichlet(w)
            t = 0
            while t < cfg.episode_len:
                vehicles.append((t, rids[rng.choice(len(rids), p=shares)], k))
                rate = q
                if cfg.demand == "peak":  # 0.6x at the start and end of the hour, 1.4x in the middle
                    rate = q * (0.6 + 0.8 * np.sin(np.pi * t / cfg.episode_len))
                t += int(rng.poisson(3600.0 / rate))
        vehicles.sort(key=lambda x: x[0])
        p_vec = np.array([cfg.p * (cfg.p_mult[k] if cfg.p_mult else 1.0) for (_, _, k) in vehicles])
        connected = rng.random(len(vehicles)) < np.minimum(p_vec, 1.0)
        vehicles = [(t, rid) for (t, rid, _) in vehicles]
        self.n_vehicles, self.n_connected = len(vehicles), int(connected.sum())

        vt = (f'accel="0.8" decel="4.5" sigma="0.5" length="5" minGap="2.5" '
              f'maxSpeed="{cfg.v_max_vehicle}" guiShape="passenger"')
        with open(self.rou_file, "w") as f:
            f.write(f'<routes>\n    <vType id="def" {vt}/>\n    <vType id="con" {vt}/>\n')
            for rid, edges in self.routes.items():
                f.write(f'    <route id="{rid}" edges="{" ".join(edges)}"/>\n')
            for k, ((t, rid), con) in enumerate(zip(vehicles, connected)):
                f.write(f'    <vehicle id="v{k}" type="{"con" if con else "def"}" route="{rid}" depart="{t}"/>\n')
            f.write("</routes>\n")

    # ------------------------------------------------------------- simulation
    def reset(self, seed: int):
        """Start a new episode generated deterministically from ``seed``."""
        rng = np.random.default_rng(seed)
        self._write_routes(rng)
        args = self._sumo_args(sumo_seed=seed % (2 ** 31))
        if self._started:
            traci.load(args[1:])
        else:
            traci.start(args)
            self._started = True
        if self.lane_links is None:
            self.lane_links = {}
            for idx, links in enumerate(traci.trafficlight.getControlledLinks(self.tl_id)):
                for (in_lane, _, _) in links:
                    self.lane_links.setdefault(in_lane, []).append(idx)
        self.green_idx = 0
        self._set_phase(self.green_phases[0])
        self._advance(self.cfg.t_green)
        self.t_decisions = 0
        return self.observe()

    def _set_phase(self, phase: int):
        traci.trafficlight.setPhase(self.tl_id, phase)
        traci.trafficlight.setPhaseDuration(self.tl_id, 1e6)  # we control timing

    def _advance(self, seconds: int):
        for _ in range(seconds):
            traci.simulationStep()

    def step(self, action: int):
        cfg = self.cfg
        switched = action != self.green_idx
        if switched:
            g = self.green_phases[self.green_idx]
            self._set_phase(g + 1)            # yellow
            self._advance(cfg.t_yellow)
            self._set_phase(g + 2)            # all red
            self._advance(cfg.t_red)
            self.green_idx = action
            self._set_phase(self.green_phases[action])
        self._advance(cfg.t_green)
        self.t_decisions += 1
        snap = self.snapshot()
        lg = self.lane_green() if cfg.reward == "stratified_cs" else None
        r = -estimate_cost(cfg.reward, snap["costs"], snap["detected"], cfg.p, cfg.prior_cost, lg) / cfg.reward_scale
        info = {
            "true_cost": float(sum(c.sum() for c in snap["costs"])),
            "n_veh": int(sum(len(c) for c in snap["costs"])),
            "switched": switched,
            "time": traci.simulation.getTime(),
        }
        done = traci.simulation.getTime() >= cfg.episode_len
        return self.observe(snap), r, done, info

    def is_cv_type(self, type_id: str) -> bool:
        return type_id == "con"

    def close(self):
        if self._started:
            traci.close()
            self._started = False

    # ------------------------------------------------------- measurement
    def snapshot(self):
        """Per-lane costs, CV masks, speeds and distances of every vehicle."""
        costs, detected, speeds, dists = [], [], [], []
        for lane in self.lanes:
            vids = traci.lane.getLastStepVehicleIDs(lane)
            length = traci.lane.getLength(lane)
            v = np.array([traci.vehicle.getSpeed(x) for x in vids], float)
            d = np.array([traci.vehicle.getTypeID(x) == "con" for x in vids], bool)
            dist = np.array([length - traci.vehicle.getLanePosition(x) for x in vids], float)
            costs.append(vehicle_cost(v, self.cfg.v_ref))
            detected.append(d)
            speeds.append(v)
            dists.append(dist)
        return {"costs": costs, "detected": detected, "speeds": speeds, "dists": dists}

    def lane_green(self):
        state = self.phase_states[self.green_phases[self.green_idx]]
        return [any(state[i].lower() == "g" for i in self.lane_links.get(lane, [])) for lane in self.lanes]

    def observe(self, snap=None):
        snap = snap or self.snapshot()
        cfg = self.cfg
        obs = np.zeros(self.obs_shape, dtype=np.float32)
        state = self.phase_states[self.green_phases[self.green_idx]]
        for l, lane in enumerate(self.lanes):
            d, v, dist = snap["detected"][l], snap["speeds"][l], snap["dists"][l]
            for vi, di in zip(v[d], dist[d]):
                if di <= cfg.con_range:
                    cell = min(int(di / cfg.cell_length), self.n_cells - 1)
                    obs[0, l, cell] = 1.0
                    obs[1, l, cell] = vi / cfg.v_max_vehicle
            if any(state[i].lower() == "g" for i in self.lane_links.get(lane, [])):
                obs[2, l, :] = 1.0
            if cfg.obs_loop:  # loop-detector count of all vehicles on the lane (about 40 fit)
                obs[3, l, :] = len(snap["speeds"][l]) / 40.0
        return obs

    # --------------------------------------------------- loop / CV counts
    def lane_counts(self, cv_only: bool, max_dist: float | None = None, lanes=None):
        """Vehicle counts per lane, from loops (all) or CVs only."""
        out = {}
        for lane in lanes or self.lanes:
            vids = traci.lane.getLastStepVehicleIDs(lane)
            length = traci.lane.getLength(lane)
            n = 0
            for x in vids:
                if cv_only and traci.vehicle.getTypeID(x) != "con":
                    continue
                if max_dist is not None and length - traci.vehicle.getLanePosition(x) > max_dist:
                    continue
                n += 1
            out[lane] = n
        return out

    def read_tripinfo(self):
        """Per-vehicle results of the last episode (needs cfg.tripinfo).

        SUMO writes the file when the simulation ends, so this closes it; the
        next reset() starts a fresh simulation.
        """
        self.close()
        rows = []
        for _, el in ET.iterparse(self.trip_file):
            if el.tag == "tripinfo":
                rows.append({
                    "type": el.get("vType"),
                    "duration": float(el.get("duration")),
                    "time_loss": float(el.get("timeLoss")),
                    "waiting": float(el.get("waitingTime")),
                    "arrived": float(el.get("arrival")) >= 0,
                })
                el.clear()
        return rows
