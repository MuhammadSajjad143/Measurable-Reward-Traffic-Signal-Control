"""Per-seed delays and seed medians used in the results text (final model fused_cs and its
comparators), for every network and scenario.

    python analysis/final_numbers.py > results/final_numbers.txt
"""

import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ps = pd.read_csv(os.path.join(ROOT, "results", "per_seed.csv"))
same = ps.train_p.isna() | np.isclose(ps.train_p.fillna(-1), ps.eval_p)
ps = ps[same]
METHODS = [("sac_auto_loopstate", "fused_cs"), ("isac_auto_loopstate", "fused_cs"), ("sac_auto_loopstate", "fused"),
           ("isac_auto_loopstate", "fused"), ("sac_auto_loopstate", "stratified"), ("isac_auto_loopstate", "stratified"),
           ("sac_auto_loopstate", "privileged"), ("isac_auto_loopstate", "privileged"),
           ("dqn_loopstate", "pressure"), ("idqn_loopstate", "pressure"), ("sac_auto_loopstate", "pressure"),
           ("isac_auto_loopstate", "pressure"), ("max_pressure_loop", "-"), ("max_pressure_cv", "-"), ("fixed_time", "-"),
           ("fu2026_qm_atsc", "-")]
for (net, sc, p), g in ps.groupby(["net", "scenario", "eval_p"]):
    lines = []
    for m, r in METHODS:
        h = g[(g.method == m) & (g.reward == r)]
        if h.empty:
            continue
        d = h.delay_all.values
        lines.append(f"  {m + '/' + r:34s} mean {d.mean():6.1f} median {np.median(d):6.1f} gap {h.delay_gap.mean():5.1f} "
                     f"unfin {h.unfinished.mean():5.1f} seeds {[round(x, 1) for x in d]}")
    if lines:
        print(f"{net} {sc} p={p}")
        print("\n".join(lines))

# scenario-paired tests of the final model under test-time conditions (seed-averaged per episode)
import glob  # noqa: E402

from scipy import stats  # noqa: E402

DIRS = {"fused_cs": "runs/fused_cs/{net}/fused_cs_p0.1_s*", "stratified": "runs/{loop}/stratified_p0.1_s*",
        "dqn_pressure": "runs/pressure/{pnet}/dqn_loopstate/pressure_p0.1_s*",
        "sac_pressure": "runs/pressure/{pnet}/sac_auto_loopstate/pressure_p0.1_s*"}


def per_episode(name, net, tag):
    if name == "max_pressure_loop":
        base = f"runs/real/{net}/baselines" if net != "1tls_3x3" else "runs/baselines"
        f = os.path.join(ROOT, base, f"eval_max_pressure_loop_{tag}.csv")
        return pd.read_csv(f).set_index("episode").delay_all if os.path.exists(f) else None
    loop = f"real/{net}/sac_auto_loopstate" if net != "1tls_3x3" else "sac_auto_loopstate"
    pat = DIRS[name].format(net="synthetic" if net == "1tls_3x3" else net, loop=loop,
                            pnet="synthetic" if net == "1tls_3x3" else net)
    fs = [os.path.join(d, f"eval_agent_{tag}.csv") for d in sorted(glob.glob(os.path.join(ROOT, pat)))]
    fs = [f for f in fs if os.path.exists(f)]
    if not fs:
        return None
    return pd.concat([pd.read_csv(f).set_index("episode").delay_all for f in fs], axis=1).mean(axis=1)


print("=== paired tests, fused_cs vs others, p=0.1 test-time scenarios (diff = fused_cs - other)")
for net in ("cologne1", "ingolstadt1", "1tls_3x3"):
    for tag in ("p0.1", "p0.1_noise0.05", "p0.1_noise0.1", "p0.1_noise0.2", "p0.1_fail1", "p0.1_scale1.3", "p0.1_scale1.6"):
        a = per_episode("fused_cs", net, tag)
        if a is None:
            continue
        out = []
        for o in ("max_pressure_loop", "dqn_pressure", "sac_pressure", "stratified"):
            b = per_episode(o, net, tag)
            if b is None:
                continue
            c = a.index.intersection(b.index)
            out.append(f"{o} {(a[c] - b[c]).mean():+6.1f} better {(a[c] < b[c]).sum():2d}/{len(c)} "
                       f"t {stats.ttest_rel(a[c], b[c]).pvalue:.4f} w {stats.wilcoxon(a[c], b[c]).pvalue:.4f}")
        print(f"{net} {tag}: " + " | ".join(out))
