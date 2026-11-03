"""Data for the fairness-diagnosis figure on the held-out Ingolstadt network (one shared test scenario).

For the minor entry road served by junction J=3, record at every decision of that junction: vehicles N,
CVs n, true cost, the stratified estimate with the pooled fallback and with the same-signal fallback, and
whether the chosen phase gives the lane green. One CSV row per (model, decision, lane).

    python analysis/diagnose_minor_road.py RUN_DIR [RUN_DIR ...]   (writes results/diagnostics/lanediag/<run>.csv)
"""
import os
import sys

import numpy as np
import pandas as pd

sys.modules["libsumo"] = None  # socket TraCI, as in evaluation
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tsc.env import EnvConfig  # noqa: E402
from tsc.multinet import MultiRealNetEnv  # noqa: E402
from tsc.realnet import _is_green  # noqa: E402
from tsc.sac import DiscreteSAC  # noqa: E402

J, SCENARIO = 3, 900_000_000 + 19
HERE = os.path.join(ROOT, "results", "diagnostics")


def run(run_dir):
    tag = os.path.basename(run_dir)
    env = MultiRealNetEnv(EnvConfig(p=0.1, reward="stratified", obs_loop=True,
                                    workdir=os.path.join(ROOT, "runs", "lanediag_sumo_" + tag)), "ingolstadt7")
    pol = []
    for k, j in enumerate(env.junctions):
        a = DiscreteSAC(j.obs_shape, j.n_actions, seed=0, device="cpu", buffer_size=1)
        a.load(f"{run_dir}/model{k}.pt")
        pol.append(a)
    jn = env.junctions[J]
    idx = [i for i, l in enumerate(jn.lanes) if l.startswith("10425609#1")]
    obs, done, rows, t = env.reset(seed=SCENARIO), False, [], 0
    while not done:
        acts = {i: pol[i].act(o, greedy=True) for i, o in obs.items()}
        if J in acts:
            snap = jn.snapshot()
            lg = jn.lane_green()
            state = jn.phase_states[acts[J]]
            for i in idx:
                c, d = snap["costs"][i], snap["detected"][i]
                if len(c) == 0:
                    continue
                det = [cc[dd] for cc, dd in zip(snap["costs"], snap["detected"])]
                pooled = np.concatenate(det)
                fb_pool = pooled.mean() if pooled.size else 0.5
                same = [cd for cd, g in zip(det, lg) if g == lg[i]]
                same = np.concatenate(same) if same else np.empty(0)
                fb_same = same.mean() if same.size else fb_pool
                own = c[d].mean() if d.any() else None
                rows.append(dict(model=tag, t=t, lane=jn.lanes[i], N=len(c), n=int(d.sum()), true=c.sum(),
                                 est_pooled=len(c) * (own if own is not None else fb_pool),
                                 est_same=len(c) * (own if own is not None else fb_same), lane_green_now=lg[i],
                                 served=any(_is_green(state[x]) for x in jn.lane_links.get(jn.lanes[i], []))))
            t += 1
        obs, _, done, _ = env.step(acts)
    env.close()
    return pd.DataFrame(rows)


if __name__ == "__main__":
    os.makedirs(os.path.join(HERE, "lanediag"), exist_ok=True)
    for rd in sys.argv[1:]:
        d = run(rd)
        d.to_csv(os.path.join(HERE, "lanediag", os.path.basename(rd) + ".csv"), index=False)
        q = d
        print(rd, "no-CV share", round((q.n == 0).mean(), 3), "est/true pooled", round(q[q.n == 0].est_pooled.sum() / q[q.n == 0].true.sum(), 2),
              "same", round(q[q.n == 0].est_same.sum() / q[q.n == 0].true.sum(), 2),
              "served noCV", round(q[q.n == 0].served.mean(), 3), "served CV", round(q[q.n > 0].served.mean(), 3))
