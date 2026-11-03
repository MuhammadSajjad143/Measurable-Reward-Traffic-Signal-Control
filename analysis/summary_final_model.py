"""Text summary of stage 7 (proposed model fused_cs: p levels, drift, unseen demand,
loop imperfections, heavier demand). Per-seed means over the 20 held-out scenarios;
'jams' counts scenarios with mean delay > 300 s.

    python analysis/summary_final_model.py
"""

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
from exp7_final_model_tests import NETS, base_dir, fused_dir, learner_dirs  # noqa: E402


def load(f):
    return pd.read_csv(f) if os.path.exists(f) else None


def summ(dirs, tag, name):
    m, g, jams = [], [], 0
    for d in dirs:
        e = load(os.path.join(d, f"eval_agent_{tag}.csv"))
        if e is None:
            continue
        m.append(e.delay_all.mean())
        g.append(e.delay_gap.mean())
        jams += int((e.delay_all > 300).sum())
    if not m:
        return f"{name:20s} --"
    return (f"{name:20s} mean {np.mean(m):6.1f}  median-seed {np.median(m):6.1f}  "
            f"seeds {[round(x, 1) for x in m]}  gap {np.mean(g):5.1f}  jams {jams}")


def base(net, b, tag):
    e = load(os.path.join(base_dir(net), f"eval_{b}_{tag}.csv"))
    if e is None:
        return f"{b:20s} --"
    return f"{b:20s} mean {e.delay_all.mean():6.1f}  gap {e.delay_gap.mean():5.1f}  jams {int((e.delay_all > 300).sum())}"


def groups(net, p, sac_pressure=False):
    ds = learner_dirs(net, p, sac_pressure)
    out = [("fused_cs", ds[:5]), ("stratified", ds[5:10]), ("dqn_pressure", ds[10:15])]
    if sac_pressure:
        out.append(("sac_pressure", ds[15:20]))
    return out


def main():
    print("=== 1  main comparison at each p")
    for net in NETS:
        for p in (0.1, 0.2, 0.4):
            print(net, f"p{p}")
            for n, ds in groups(net, p, net != "1tls_3x3"):
                print("  ", summ(ds, f"p{p}", n))
            for b in ("fixed_time", "max_pressure_cv", "max_pressure_loop"):
                print("  ", base(net, b, f"p{p}"))
    print("=== 3  drift (fused_cs): rows train p, cols eval p 0.1/0.2/0.4/1.0, mean/median-seed")
    for net in NETS:
        print(net)
        for p in (0.1, 0.2, 0.4):
            row = []
            for q in (0.1, 0.2, 0.4, 1.0):
                v = [load(os.path.join(fused_dir(net, p, s), f"eval_agent_p{q}.csv")) for s in range(5)]
                v = [x.delay_all.mean() for x in v if x is not None]
                row.append(f"{np.mean(v):6.1f}/{np.median(v):6.1f}" if v else "   --  ")
            print(f"  train {p}: " + "  ".join(row))
    print("=== 4  unseen demand, synthetic fused_cs")
    for dem in ("fixed", "peak"):
        for p in (0.1, 0.2, 0.4):
            v = [load(os.path.join(fused_dir("1tls_3x3", p, s), f"eval_agent_p{p}_demand_{dem}.csv")) for s in range(5)]
            v = [x.delay_all.mean() for x in v if x is not None]
            print(f"  {dem} p{p}: {np.mean(v):6.1f} / {np.median(v):6.1f} {[round(x, 1) for x in v]}")
    print("=== 5  loop imperfections, p = 0.1")
    for net in NETS:
        for tag in ("p0.1", "p0.1_noise0.05", "p0.1_noise0.1", "p0.1_noise0.2", "p0.1_fail1"):
            print(net, tag)
            for n, ds in groups(net, 0.1):
                print("  ", summ(ds, tag, n))
            print("  ", base(net, "max_pressure_loop", tag))
    print("=== 6  heavier demand, p = 0.1")
    for net in ("cologne1", "ingolstadt1"):
        for tag in ("p0.1", "p0.1_scale1.3", "p0.1_scale1.6"):
            print(net, tag)
            for n, ds in groups(net, 0.1, True):
                print("  ", summ(ds, tag, n))
            for b in ("fixed_time", "max_pressure_cv", "max_pressure_loop"):
                print("  ", base(net, b, tag))


if __name__ == "__main__":
    main()
