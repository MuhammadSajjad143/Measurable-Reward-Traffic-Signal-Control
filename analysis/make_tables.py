"""Write the paper's LaTeX tables from the analysis outputs (results/) and training logs.

Every number in the tables comes from these files; rerun after analysis/analyze.py.

    python analysis/make_tables.py
"""

from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(RES, "tables")
P_LEVELS = (0.1, 0.2, 0.4)
REWARDS = ("privileged", "naive", "ipw", "loop_count", "stratified")
R_NAME = {"privileged": "Privileged$^\\dagger$", "naive": "Naive (CV-only)", "ipw": "IPW",
          "loop_count": "Loop count", "stratified": "Lane-stratified", "pressure": "Pressure",
          "fused": "Fused, pooled fallback", "fused_cs": "\\textbf{Fused (proposed)}"}
BASE = [("fixed_time", "-", "Fixed time"), ("max_pressure_cv", "-", "Max-pressure (CV counts)"),
        ("max_pressure_loop", "-", "Max-pressure (loop counts)"), ("fu2026_qm_atsc", "-", "QM-ATSC \\citep{fu2026}")]


def fmt(m, c, d=1):
    if pd.isna(m):
        return "--"
    return f"{m:.{d}f}" if pd.isna(c) else f"{m:.{d}f} $\\pm$ {c:.{d}f}"


def main_rows(s, net):
    same = s.train_p.isna() | np.isclose(s.train_p.fillna(-1), s.eval_p)
    return s[(s.net == net) & (s.scenario == "main") & same]


def cell(g, method, reward, p, metric):
    r = g[(g.method == method) & (g.reward == reward) & np.isclose(g.eval_p, p)]
    if r.empty:
        return "--"
    r = r.iloc[0]
    return fmt(r[metric], r[metric + "_ci"])


def delay_table(s, net, groups, caption, label, path, ps=P_LEVELS, seeds_note=""):
    g = main_rows(s, net)
    cols = "l" + "cc" * len(ps)
    lines = ["\\begin{table}[t]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\scriptsize",
             "\\setlength{\\tabcolsep}{3pt}", f"\\begin{{tabular}}{{@{{}}{cols}@{{}}}}", "\\toprule",
             " & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{$p={p}$}}" for p in ps) + "\\\\",
             "".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(ps))),
             "Controller & " + " & ".join("Delay (s) & Gap (s)" for _ in ps) + "\\\\", "\\midrule"]
    for title, rows in groups:
        rows = [(m, r, n) for m, r, n in rows if not g[(g.method == m) & (g.reward == r)].empty]
        if not rows:
            continue
        if title:
            lines.append(f"\\multicolumn{{{1 + 2 * len(ps)}}}{{@{{}}l}}{{\\textit{{{title}}}}}\\\\")
        for m, r, name in rows:
            vals = []
            for p in ps:
                vals += [cell(g, m, r, p, "delay_all"), cell(g, m, r, p, "delay_gap")]
            lines.append(f"{name} & " + " & ".join(vals) + "\\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "", "\\vspace{2pt}",
              "{\\footnotesize Mean $\\pm$ 95\\% $t$-interval across training seeds; each seed is averaged over the same 20 "
              "held-out scenarios. Gap = delay of unconnected minus connected vehicles. $^\\dagger$Not measurable at a real "
              f"intersection (upper reference).{seeds_note}}}", "\\end{table}"]
    open(path, "w").write("\n".join(lines) + "\n")


def sac_groups(extra=()):
    return [("Baselines", BASE),
            ("SAC, CV-only state", [("sac_auto", r, R_NAME[r]) for r in REWARDS]),
            ("SAC, loop--CV state", [("sac_auto_loopstate", r, R_NAME[r]) for r in REWARDS + ("fused", "fused_cs")]),
            *extra]


PRESSURE = ("Pressure reward, loop--CV state", [("dqn_loopstate", "pressure", "PressLight-style DQN \\citep{wei2019}"),
                                                ("sac_auto_loopstate", "pressure", "SAC, pressure reward")])


def estimator_table():
    e = pd.read_csv(os.path.join(RES, "estimator_study.csv"))
    ps = [0.1, 0.2, 0.4]
    lines = ["\\begin{table}[t]", "\\centering",
             "\\caption{Accuracy of the measurable cost estimates on recorded SUMO snapshots (no learning). Relative bias "
             "and relative RMSE with respect to the true cost; phase agreement = share of snapshots in which the estimate "
             "ranks the same phase as most congested as the true cost.}",
             "\\label{tab:estimators_acc}", "\\scriptsize", "\\setlength{\\tabcolsep}{3pt}", "\\begin{tabular}{@{}l" + "ccc" * len(ps) + "@{}}", "\\toprule",
             " & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{$p={p}$}}" for p in ps) + "\\\\",
             "".join(f"\\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(len(ps))),
             "Estimator & " + " & ".join("Bias & RMSE & Agree." for _ in ps) + "\\\\", "\\midrule"]
    for est in ("naive", "ipw", "loop_count", "stratified"):
        vals = []
        for p in ps:
            r = e[(e.estimator == est) & np.isclose(e.p, p)].iloc[0]
            vals += [f"{100 * r.rel_bias:+.1f}\\%", f"{100 * r.rel_rmse:.1f}\\%", f"{100 * r.phase_agreement:.0f}\\%"]
        lines.append(f"{R_NAME[est]} & " + " & ".join(vals) + "\\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    open(os.path.join(OUT, "estimators.tex"), "w").write("\n".join(lines) + "\n")


def robustness_table():
    r = pd.read_csv(os.path.join(RES, "estimator_robustness.csv"))
    r.to_csv(os.path.join(OUT, "robustness_raw.csv"), index=False)  # for the text; kept beside the tables


def temperature_table():
    """Final policy entropy and temperature (last 10% of training), fixed vs automatic temperature."""
    rows = []
    for tag in ("sac_alpha0.2", "sac_auto"):
        for f in glob.glob(os.path.join(ROOT, "runs", tag, "*", "train_log.csv")):
            name = os.path.basename(os.path.dirname(f))
            reward, p, seed = name.rsplit("_", 2)
            log = pd.read_csv(f).dropna(subset=["entropy"])
            if log.empty or not os.path.exists(os.path.join(os.path.dirname(f), "DONE")):
                continue
            late = log[log.step > 0.9 * log.step.max()]
            rows.append({"tag": tag, "reward": reward, "p": float(p[1:]), "entropy": late.entropy.mean(),
                         "alpha": late.alpha.mean()})
    d = pd.DataFrame(rows).groupby(["tag", "reward", "p"]).agg(entropy=("entropy", "mean"), alpha=("alpha", "mean"),
                                                              n=("entropy", "size")).reset_index()
    d.to_csv(os.path.join(OUT, "temperature_raw.csv"), index=False)
    ps = P_LEVELS
    lines = ["\\begin{table}[t]", "\\centering",
             "\\caption{Final policy entropy (nats) with a fixed temperature $\\alpha=0.2$ and learned temperature "
             "$\\alpha$ with automatic tuning (mean of the last 10\\% of training, CV-only state). The last row is the "
             "ratio of the learned temperatures of the naive and IPW rewards, which Corollary~\\ref{cor:temp} predicts "
             "to be about $p$.}", "\\label{tab:temperature}", "\\small",
             "\\begin{tabular}{@{}l" + "c" * len(ps) + "@{}}", "\\toprule",
             "& " + " & ".join(f"$p={p}$" for p in ps) + "\\\\", "\\midrule",
             f"\\multicolumn{{{1 + len(ps)}}}{{@{{}}l}}{{\\textit{{Entropy, fixed $\\alpha=0.2$}}}}\\\\"]
    get = lambda tag, r, p, k: d[(d.tag == tag) & (d.reward == r) & np.isclose(d.p, p)][k]
    for r in ("privileged", "naive", "ipw"):
        lines.append(f"{R_NAME[r]} & " + " & ".join(f"{get('sac_alpha0.2', r, p, 'entropy').iloc[0]:.2f}"
                                                    if len(get('sac_alpha0.2', r, p, 'entropy')) else "--" for p in ps) + "\\\\")
    lines.append(f"\\multicolumn{{{1 + len(ps)}}}{{@{{}}l}}{{\\textit{{Learned temperature, automatic tuning}}}}\\\\")
    for r in ("privileged", "naive", "ipw", "stratified"):
        lines.append(f"{R_NAME[r]} & " + " & ".join(f"{get('sac_auto', r, p, 'alpha').iloc[0]:.4f}" for p in ps) + "\\\\")
    ratio = [get("sac_auto", "naive", p, "alpha").iloc[0] / get("sac_auto", "ipw", p, "alpha").iloc[0] for p in ps]
    lines.append("Ratio naive / IPW & " + " & ".join(f"{x:.2f}" for x in ratio) + "\\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    open(os.path.join(OUT, "temperature.tex"), "w").write("\n".join(lines) + "\n")


def transfer_table(ps):
    tests = P_LEVELS + (1.0,)
    rows = [("1tls_3x3", "sac_auto", "stratified", "Synthetic", "CV-only state, stratified"),
            ("1tls_3x3", "sac_auto_loopstate", "stratified", "", "Loop--CV state, stratified"),
            ("1tls_3x3", "sac_auto_loopstate", "fused_cs", "", "\\textbf{Proposed}"),
            ("cologne1", "sac_auto_loopstate", "fused_cs", "Cologne", "\\textbf{Proposed}"),
            ("ingolstadt1", "sac_auto_loopstate", "fused_cs", "Ingolstadt", "\\textbf{Proposed}")]
    lines = ["\\begin{table}[t]", "\\centering",
             "\\caption{Penetration drift: mean delay (s) when a model trained at one $p$ is tested at another. "
             "Diagonal = test $p$ equals training $p$.}",
             "\\label{tab:transfer}", "\\scriptsize", "\\setlength{\\tabcolsep}{3pt}",
             "\\begin{tabular}{@{}lll" + "c" * len(tests) + "@{}}", "\\toprule",
             "Network & Controller & Train $p$ & " + " & ".join(f"Test {p}" for p in tests) + "\\\\", "\\midrule"]
    for i, (net, m, r, nname, name) in enumerate(rows):
        if i and nname:
            lines.append("\\midrule")
        elif i:
            lines.append("\\cmidrule(l){2-7}")
        for tp in P_LEVELS:
            vals = [seed_cell(ps, net, m, r, tp, ep) for ep in tests]
            lines.append(f"{nname if tp == P_LEVELS[0] else ''} & {name if tp == P_LEVELS[0] else ''} & {tp} & "
                         + " & ".join(vals) + "\\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "", "\\vspace{2pt}",
              "{\\footnotesize Mean over five training seeds; the median over seeds is given in parentheses when it differs "
              "from the mean by more than 10\\%.}", "\\end{table}"]
    open(os.path.join(OUT, "transfer.tex"), "w").write("\n".join(lines) + "\n")


def scenario_table(s):
    """Fixed and peak-hour demand and uneven penetration (synthetic), stratified vs baselines."""
    rows = [*BASE, ("sac_auto", "stratified", "SAC, CV-only state, lane-stratified"),
            ("sac_auto_loopstate", "stratified", "SAC, loop--CV state, lane-stratified"),
            ("sac_auto_loopstate", "fused_cs", "\\textbf{SAC, loop--CV, fused (proposed)}")]
    scen = [("demand_fixed", "Fixed demand"), ("demand_peak", "Peak hour"), ("uneven", "Uneven penetration")]
    g = s[(s.net == "1tls_3x3")]
    same = g.train_p.isna() | np.isclose(g.train_p.fillna(-1), g.eval_p)
    g = g[same]
    lines = ["\\begin{table}[t]", "\\centering",
             "\\caption{Generalisation beyond the training conditions (synthetic intersection): mean delay (s) under "
             "demand patterns not seen in training and with uneven penetration by approach.}",
             "\\label{tab:scenarios}", "\\scriptsize", "\\setlength{\\tabcolsep}{3pt}",
             "\\begin{tabular}{@{}ll" + "ccc" + "@{}}", "\\toprule",
             "Scenario & Controller & $p=0.1$ & $p=0.2$ & $p=0.4$\\\\", "\\midrule"]
    for sc, sname in scen:
        first = True
        for m, r, name in rows:
            h = g[(g.scenario == sc) & (g.method == m) & (g.reward == r)]
            if h.empty:
                continue
            vals = [fmt(*h[np.isclose(h.eval_p, p)][["delay_all", "delay_all_ci"]].iloc[0])
                    if (np.isclose(h.eval_p, p)).any() else "--" for p in P_LEVELS]
            lines.append(f"{sname if first else ''} & {name} & " + " & ".join(vals) + "\\\\")
            first = False
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines += ["\\end{tabular}", "\\end{table}"]
    open(os.path.join(OUT, "scenarios.tex"), "w").write("\n".join(lines) + "\n")


def sensitivity_table(s):
    g = s[(s.net == "1tls_3x3") & (s.scenario == "main") & (s.reward == "stratified")
          & np.isclose(s.train_p.fillna(-1), 0.2) & np.isclose(s.eval_p, 0.2)]
    rows = [("sac_auto_loopstate", "Main setting ($\\bar H = 0.5\\log|\\mathcal{A}|$, $K=50$)"),
            ("sac_auto_loopstate_H0.3", "Target entropy ratio 0.3"), ("sac_auto_loopstate_H0.7", "Target entropy ratio 0.7"),
            ("sac_auto_loopstate_K25", "Reward scale $K=25$"), ("sac_auto_loopstate_K100", "Reward scale $K=100$")]
    lines = ["\\begin{table}[t]", "\\centering",
             "\\caption{Sensitivity of SAC with the loop--CV state and the lane-stratified reward (synthetic intersection, "
             "$p=0.2$). Five seeds for the main setting, three for each variant.}", "\\label{tab:sensitivity}", "\\small", "\\begin{tabular}{@{}lcc@{}}", "\\toprule",
             "Setting & Delay (s) & Gap (s)\\\\", "\\midrule"]
    for m, name in rows:
        r = g[g.method == m]
        if r.empty:
            continue
        r = r.iloc[0]
        lines.append(f"{name} & {fmt(r.delay_all, r.delay_all_ci)} & {fmt(r.delay_gap, r.delay_gap_ci)}\\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    open(os.path.join(OUT, "sensitivity.tex"), "w").write("\n".join(lines) + "\n")


def multi_table(s):
    groups = [("Baselines (per signal)", BASE[:3]),
              ("Independent SAC, loop--CV state", [("isac_auto_loopstate", r, R_NAME[r]) for r in REWARDS + ("fused", "fused_cs")]),
              ("Pressure reward", [("idqn_loopstate", "pressure", "PressLight-style DQN \\citep{wei2019}"),
                                   ("isac_auto_loopstate", "pressure", "SAC, pressure reward")])]
    delay_table(s, "cologne3", groups,
                "Real three-signal network (RESCO Cologne, 07:00--08:00): network-wide delay and gap.",
                "tab:multi", os.path.join(OUT, "multi.tex"), ps=(0.1, 0.2), seeds_note=" Three seeds per controller.")


def heldout_table(s):
    nets = [("ingolstadt7", "Ingolstadt, 7 signals"), ("cologne8", "Cologne, 8 signals")]
    rows = [("Baselines (per signal)", BASE[:3]),
            ("Learned, one agent per signal, loop--CV state", [
                ("isac_auto_loopstate", "stratified", "SAC, lane-stratified"),
                ("isac_auto_loopstate", "fused", "SAC, fused, pooled fallback"),
                ("isac_auto_loopstate", "fused_cs", "\\textbf{SAC, fused (proposed)}"),
                ("idqn_loopstate", "pressure", "PressLight-style DQN")])]
    lines = ["\\begin{table}[t]", "\\centering",
             "\\caption{Held-out real networks (not used to choose the reward weight or any hyperparameter): network-wide delay and gap at $p=0.1$.}",
             "\\label{tab:heldout}", "\\scriptsize", "\\setlength{\\tabcolsep}{3pt}", "\\begin{tabular}{@{}lcccc@{}}",
             "\\toprule", " & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{n}}}" for _, n in nets) + "\\\\",
             "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}", "Controller & Delay (s) & Gap (s) & Delay (s) & Gap (s)\\\\", "\\midrule"]
    for title, group in rows:
        lines.append(f"\\multicolumn{{5}}{{@{{}}l}}{{\\textit{{{title}}}}}\\\\")
        for m, r, name in group:
            vals = []
            for net, _ in nets:
                g = main_rows(s, net)
                vals += [cell(g, m, r, 0.1, "delay_all"), cell(g, m, r, 0.1, "delay_gap")]
            lines.append(f"{name} & " + " & ".join(vals) + "\\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "", "\\vspace{2pt}",
              "{\\footnotesize Mean $\\pm$ 95\\% $t$-interval across three training seeds; each seed is averaged over the "
              "same 20 held-out scenarios. Gap = delay of unconnected minus connected vehicles. RESCO scenarios with "
              "calibrated demand: Ingolstadt 16:00--17:00, Cologne 07:00--08:00.}", "\\end{table}"]
    open(os.path.join(OUT, "heldout.tex"), "w").write("\n".join(lines) + "\n")


def seed_cell(ps, net, method, reward, train_p, eval_p, scenario="main"):
    """Mean delay over seeds, with the median in parentheses when it differs from the mean by more than 10%."""
    h = ps[(ps.net == net) & (ps.method == method) & (ps.reward == reward) & (ps.scenario == scenario)
           & np.isclose(ps.eval_p, eval_p)]
    if train_p is not None:
        h = h[np.isclose(h.train_p, train_p)]
    if h.empty:
        return "--"
    m, med = h.delay_all.mean(), h.delay_all.median()
    return f"{m:.1f}" + (f" ({med:.1f})" if abs(m - med) > 0.1 * med else "")


def testtime_table(ps):
    """Loop-detector imperfections and heavier demand at test time (p = 0.1, models trained at p = 0.1)."""
    cols = [("main", "Nominal"), ("noise0.05", "5\\%"), ("noise0.1", "10\\%"), ("noise0.2", "20\\%"),
            ("fail1", "Failed loop"), ("scale1.3", "$\\times1.3$"), ("scale1.6", "$\\times1.6$")]
    rows = [("sac_auto_loopstate", "fused_cs", "\\textbf{Proposed}"),
            ("sac_auto_loopstate", "stratified", "SAC, lane-stratified"),
            ("dqn_loopstate", "pressure", "PressLight-style DQN"), ("sac_auto_loopstate", "pressure", "SAC, pressure reward"),
            ("max_pressure_loop", "-", "Max-pressure (loops)")]
    nets = [("cologne1", "Cologne"), ("ingolstadt1", "Ingolstadt"), ("1tls_3x3", "Synthetic")]
    lines = ["\\begin{table}[t]", "\\centering",
             "\\caption{Test-time robustness at $p=0.1$: mean delay (s) with noisy loop counts (relative standard "
             "deviation 5--20\\%), one failed loop detector that reports zero, and calibrated demand scaled by 1.3 and "
             "1.6. Models are trained at $p=0.1$ with exact counts and nominal demand.}", "\\label{tab:testtime}",
             "\\scriptsize", "\\setlength{\\tabcolsep}{2.5pt}", "\\begin{tabular}{@{}l" + "c" * len(cols) + "@{}}", "\\toprule",
             " & & \\multicolumn{3}{c}{Loop-count noise} & & \\multicolumn{2}{c}{Demand}\\\\",
             "\\cmidrule(lr){3-5}\\cmidrule(lr){7-8}", "Controller & " + " & ".join(n for _, n in cols) + "\\\\", "\\midrule"]
    for net, nname in nets:
        lines.append(f"\\multicolumn{{{1 + len(cols)}}}{{@{{}}l}}{{\\textit{{{nname}}}}}\\\\")
        for m, r, name in rows:
            vals = [seed_cell(ps, net, m, r, None if r == "-" else 0.1, 0.1, sc) for sc, _ in cols]
            if all(v == "--" for v in vals[1:]) and net == "1tls_3x3":
                continue
            lines.append(f"{name} & " + " & ".join(vals) + "\\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "", "\\vspace{2pt}",
              "{\\footnotesize Mean over five training seeds (one run for max-pressure), each averaged over the same 20 "
              "held-out scenarios; the median over seeds is given in parentheses when it differs from the mean by more "
              "than 10\\%. The synthetic intersection has no calibrated demand to scale.}", "\\end{table}"]
    open(os.path.join(OUT, "testtime.tex"), "w").write("\n".join(lines) + "\n")


def compact_table(s):
    """Main comparison at p = 0.1 on the three single intersections (Springer version, main text)."""
    nets = [("1tls_3x3", "Synthetic"), ("cologne1", "Cologne"), ("ingolstadt1", "Ingolstadt")]
    base = [("fixed_time", "-", "Fixed time"), ("max_pressure_cv", "-", "Max-pressure, CV"),
            ("max_pressure_loop", "-", "Max-pressure, loops"), ("fu2026_qm_atsc", "-", "QM-ATSC \\citep{fu2026}")]
    names = dict(R_NAME, naive="Naive", fused="Fused, pooled", fused_cs="\\textbf{Fused (proposed)}")
    groups = [("Classical and tabular baselines", base),
              ("SAC, CV-only state", [("sac_auto", "stratified", "Lane-stratified")]),
              ("SAC, loop--CV state", [("sac_auto_loopstate", r, names[r]) for r in REWARDS + ("fused", "fused_cs")]),
              ("Pressure reward, loop--CV state", [("dqn_loopstate", "pressure", "PressLight-style DQN"),
                                                   ("sac_auto_loopstate", "pressure", "SAC, pressure")])]
    f2 = lambda m, c: "--" if pd.isna(m) else (f"{m:.1f}" if pd.isna(c) else f"{m:.1f} ({c:.1f})")
    lines = ["\\begin{table}[t]", "\\caption{Mean delay and delay gap (s) at $p=0.1$ on the single intersections: mean "
             "and, in parentheses, half-width of the 95\\% $t$-interval over training seeds. Results for $p=0.2$ and "
             "$0.4$ are in Fig.~\\ref{fig:delaygap} and Online Resource~1.}", "\\label{tab:main}", "\\scriptsize",
             "\\setlength{\\tabcolsep}{2.4pt}", "\\begin{tabular*}{\\textwidth}{@{\\extracolsep\\fill}lcccccc@{}}", "\\toprule",
             " & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{n}}}" for _, n in nets) + "\\\\",
             "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}",
             "Controller & " + " & ".join("Delay & Gap" for _ in nets) + "\\\\", "\\midrule"]
    for title, rows in groups:
        lines.append(f"\\multicolumn{{7}}{{@{{}}l}}{{\\textit{{{title}}}}}\\\\")
        for m, r, name in rows:
            vals = []
            for net, _ in nets:
                g = main_rows(s, net)
                h = g[(g.method == m) & (g.reward == r) & np.isclose(g.eval_p, 0.1)]
                vals += ["--", "--"] if h.empty else [f2(h.delay_all.iloc[0], h.delay_all_ci.iloc[0]),
                                                       f2(h.delay_gap.iloc[0], h.delay_gap_ci.iloc[0])]
            lines.append(f"{name} & " + " & ".join(vals) + "\\\\")
    lines += ["\\botrule", "\\end{tabular*}", "\\footnotetext{Gap: delay of unconnected minus connected vehicles. "
              "$^\\dagger$Not measurable at a real intersection (upper reference). Five seeds for the loop--CV state "
              "and the pressure reward, three for the CV-only state on the real intersections.}", "\\end{table}"]
    open(os.path.join(OUT, "main_p01.tex"), "w").write("\n".join(lines) + "\n")


def ablation_table(s):
    """Effect of removing or replacing each component of the proposed controller (p = 0.1)."""
    nets = [("1tls_3x3", "Synthetic"), ("cologne1", "Cologne"), ("ingolstadt1", "Ingolstadt")]
    full = ("sac_auto_loopstate", "fused_cs")
    rows = [(full, "Full model (SAC, loop--CV state, fused reward, same-signal fallback)"),
            (("sac_auto_loopstate", "fused"), "Same-signal $\\to$ pooled fallback"),
            (("sac_auto_loopstate", "stratified"), "Without the pressure term (and pooled fallback)"),
            (("sac_auto_loopstate", "naive"), "Stratified cost $\\to$ CV-only (naive) cost"),
            (("sac_auto", "stratified"), "Without the loop-count state channel$^a$"),
            (("sac_auto_loopstate", "pressure"), "Fused reward $\\to$ pressure only"),
            (("dqn_loopstate", "pressure"), "Pressure only, SAC $\\to$ double DQN")]
    get = lambda g, k: g[(g.method == k[0]) & (g.reward == k[1]) & np.isclose(g.eval_p, 0.1)]
    lines = ["\\begin{table}[t]", "\\caption{Ablation study at $p=0.1$: mean delay (s), with the change relative to the full "
             "model in parentheses, and delay gap (s). Means over five seeds (three for the CV-only state on the real "
             "intersections)}", "\\label{tab:ablation}", "\\scriptsize", "\\setlength{\\tabcolsep}{2.4pt}",
             "\\begin{tabular*}{\\textwidth}{@{\\extracolsep\\fill}>{\\raggedright\\arraybackslash}p{4.3cm}cccccc@{}}", "\\toprule",
             " & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{n}}}" for _, n in nets) + "\\\\",
             "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}",
             "Variant & " + " & ".join("Delay & Gap" for _ in nets) + "\\\\", "\\midrule"]
    for k, name in rows:
        vals = []
        for net, _ in nets:
            g = main_rows(s, net)
            h, f = get(g, k), get(g, full)
            if h.empty:
                vals += ["--", "--"]
                continue
            d, d0 = h.delay_all.iloc[0], f.delay_all.iloc[0]
            vals += [f"{d:.1f}" if k == full else f"{d:.1f} ({d - d0:+.1f})", f"{h.delay_gap.iloc[0]:.1f}"]
        lines.append(f"{name} & " + " & ".join(vals) + "\\\\")
    lines += ["\\botrule", "\\end{tabular*}", "\\footnotetext{$^a$CV-only state with the stratified reward (the fused "
              "reward was not trained with the CV-only state).}", "\\end{table}"]
    open(os.path.join(OUT, "ablation.tex"), "w").write("\n".join(lines) + "\n")


def basepaper_table(s):
    """Proposed controller against the re-implemented base methods (p = 0.1), with relative improvement."""
    nets = [("1tls_3x3", "Synthetic"), ("cologne1", "Cologne"), ("ingolstadt1", "Ingolstadt")]
    prop = ("sac_auto_loopstate", "fused_cs")
    rows = [(("fu2026_qm_atsc", "-"), "QM-ATSC, Fu et al.\\ \\citep{fu2026}$^\\dagger$"),
            (("dqn", "privileged"), "DQN, CV-only state, all-vehicle reward \\citep{ducrocq2023}$^\\dagger$"),
            (("dqn_loopstate", "pressure"), "PressLight-style DQN \\citep{wei2019}$^\\dagger$"),
            (("max_pressure_loop", "-"), "Max-pressure, loop counts \\citep{varaiya2013}$^\\dagger$")]
    get = lambda g, k: g[(g.method == k[0]) & (g.reward == k[1]) & np.isclose(g.eval_p, 0.1)]
    lines = ["\\begin{table}[t]", "\\caption{Quantitative comparison with the base methods at $p=0.1$ in the same setting "
             "(same networks, 20 test scenarios and metric): mean delay (s) and relative change of the proposed controller "
             "(negative: proposed is faster)}", "\\label{tab:basecmp}", "\\scriptsize", "\\setlength{\\tabcolsep}{2.4pt}",
             "\\begin{tabular*}{\\textwidth}{@{\\extracolsep\\fill}>{\\raggedright\\arraybackslash}p{4.6cm}cccccc@{}}", "\\toprule",
             " & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{n}}}" for _, n in nets) + "\\\\",
             "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}",
             "Method & " + " & ".join("Delay & Change" for _ in nets) + "\\\\", "\\midrule"]
    for k, name in rows + [(prop, "\\textbf{Proposed}")]:
        vals = []
        for net, _ in nets:
            g = main_rows(s, net)
            h, f = get(g, k), get(g, prop)
            if h.empty:
                vals += ["--", "--"]
                continue
            d, d0 = h.delay_all.iloc[0], f.delay_all.iloc[0]
            vals += [f"{d:.1f}", "--" if k == prop else f"{100 * (d0 - d) / d:+.1f}\\%"]
        lines.append(f"{name} & " + " & ".join(vals) + "\\\\")
    lines += ["\\botrule", "\\end{tabular*}", "\\footnotetext{$^\\dagger$Re-implemented in the present setting (QM-ATSC "
              "adapted to the phases used here; DQN of the base code with the cost of Eq.~\\ref{eq:cost} as the "
              "all-vehicle reward, trained on the synthetic intersection only; PressLight-style DQN with the proposed "
              "state and count-based pressure). Five seeds per learner, three for the DQN of \\citep{ducrocq2023}.}", "\\end{table}"]
    open(os.path.join(OUT, "basecmp.tex"), "w").write("\n".join(lines) + "\n")


def main():
    os.makedirs(OUT, exist_ok=True)
    s = pd.read_csv(os.path.join(RES, "summary.csv"))
    estimator_table()
    robustness_table()
    temperature_table()
    delay_table(s, "1tls_3x3", sac_groups([PRESSURE]),
                "Synthetic intersection: mean delay and delay gap at the training penetration rate.",
                "tab:synthetic", os.path.join(OUT, "synthetic.tex"), seeds_note=" Five seeds per learned controller.")
    for net, name in (("cologne1", "Cologne"), ("ingolstadt1", "Ingolstadt")):
        delay_table(s, net, sac_groups([PRESSURE]),
                    f"Real intersection ({name}, RESCO): mean delay and delay gap at the training penetration rate.",
                    f"tab:{net}", os.path.join(OUT, f"{net}.tex"),
                    seeds_note=" Five seeds for the loop--CV state and the pressure reward; three for the CV-only state.")
    ps = pd.read_csv(os.path.join(RES, "per_seed.csv"))
    transfer_table(ps)
    testtime_table(ps)
    scenario_table(s)
    sensitivity_table(s)
    compact_table(s)
    ablation_table(s)
    basepaper_table(s)
    if (s.net == "cologne3").any():
        multi_table(s)
    if (s.net == "ingolstadt7").any():
        heldout_table(s)
    print("tables written to", OUT)


if __name__ == "__main__":
    main()
