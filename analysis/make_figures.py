"""Write the paper's result figures (PDF, vector) from the analysis outputs (results/) and training logs.

Every value plotted comes from these files; rerun after scripts/analyze.py.
Sizes follow the Springer Nature template (text width 372 pt = 5.16 in).

    python analysis/make_figures.py
"""

from __future__ import annotations

import glob
import os
import xml.etree.ElementTree as ET

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "figures")
TW = 5.16  # text width (in)
P_LEVELS = (0.1, 0.2, 0.4)

# one colour per controller in every figure; at most three colours share a panel,
# everything else is drawn in grey with its own line style and marker
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
GREY1, GREY2 = "#6b6a66", "#a3a29d"
STYLE = {
    "proposed": dict(color=BLUE, marker="o", ls="-", label="Proposed (SAC, fused reward)"),
    "mp_loop": dict(color=ORANGE, marker="s", ls="-", label="Max-pressure (loop counts)"),
    "dqn_pressure": dict(color=AQUA, marker="^", ls="-", label="PressLight-style DQN"),
    "stratified": dict(color=GREY1, marker="D", ls="--", label="SAC, lane-stratified"),
    "cv_only": dict(color=GREY1, marker="v", ls=":", label="SAC, CV-only state"),
    "mp_cv": dict(color=GREY2, marker="x", ls="-.", label="Max-pressure (CV counts)"),
}
NETS = [("1tls_3x3", "Synthetic"), ("cologne1", "Cologne"), ("ingolstadt1", "Ingolstadt")]

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Liberation Sans", "DejaVu Sans"],
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "lines.linewidth": 1.4,
    "lines.markersize": 4.5, "legend.frameon": False, "pdf.fonttype": 42, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02, "axes.titleweight": "bold", "axes.titlelocation": "left",
})

# (net, method, reward) of each controller in summary.csv / per_seed.csv
KEY = {"proposed": ("sac_auto_loopstate", "fused_cs"), "stratified": ("sac_auto_loopstate", "stratified"),
       "dqn_pressure": ("dqn_loopstate", "pressure"), "cv_only": ("sac_auto", "stratified"),
       "mp_loop": ("max_pressure_loop", "-"), "mp_cv": ("max_pressure_cv", "-")}


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=200)
    plt.close(fig)


def panel_label(ax, k, title):
    ax.set_title(f"({'abcdefghij'[k]}) {title}")


def main_rows(s, net):
    same = s.train_p.isna() | np.isclose(s.train_p.fillna(-1), s.eval_p)
    return s[(s.net == net) & (s.scenario == "main") & same]


# --------------------------------------------------------------------------------------------
def fig_estimators():
    e = pd.read_csv(os.path.join(RES, "estimator_study.csv"))
    e = e[e.p < 1.0]
    sty = {"stratified": dict(color=BLUE, marker="o", ls="-", label="Lane-stratified"),
           "ipw": dict(color=INK2, marker="s", ls="-", label="IPW"),
           "naive": dict(color=GREY1, marker="^", ls=":", label="Naive (CV-only)"),
           "loop_count": dict(color=GREY2, marker="D", ls="--", label="Loop count")}
    fig, axs = plt.subplots(1, 3, figsize=(TW, 1.95))
    for k, (col, title, scale) in enumerate((("rel_bias", "Relative bias (%)", 100), ("rel_rmse", "Relative RMSE (%)", 100),
                                             ("phase_agreement", "Phase agreement (%)", 100))):
        ax = axs[k]
        for est, st in sty.items():
            g = e[e.estimator == est].sort_values("p")
            ax.plot(g.p, scale * g[col], **st)
        if col == "rel_bias":
            pp = np.linspace(0.05, 0.8, 50)
            ax.plot(pp, 100 * (pp - 1), color=INK2, lw=0.8, ls=(0, (1, 1.5)), label="Theory: $p-1$")
        if col == "rel_rmse":
            ax.set_ylim(0, 60)
        ax.set_xlabel("Penetration rate $p$")
        panel_label(ax, k, title)
    h, lab = axs[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig_estimators")


def fig_temperature():
    d = pd.read_csv(os.path.join(RES, "tables", "temperature_raw.csv"))
    fig, axs = plt.subplots(1, 2, figsize=(TW * 0.78, 1.95))
    sty = {"naive": dict(color=GREY1, marker="^", ls=":", label="Naive (CV-only)"), "ipw": dict(color=INK2, marker="s", label="IPW"),
           "privileged": dict(color=GREY2, marker="D", ls="--", label="Privileged")}
    for r, st in sty.items():
        g = d[(d.tag == "sac_alpha0.2") & (d.reward == r)].sort_values("p")
        axs[0].plot(g.p, g.entropy, **st)
    axs[0].set_xlabel("Penetration rate $p$")
    axs[0].set_ylabel("Policy entropy (nats)")
    axs[0].legend(loc="upper right")
    panel_label(axs[0], 0, "Fixed temperature $\\alpha=0.2$")
    a = d[d.tag == "sac_auto"].pivot(index="p", columns="reward", values="alpha")
    ratio = a["naive"] / a["ipw"]
    pp = np.linspace(0.08, 0.45, 20)
    axs[1].plot(pp, pp, color=INK2, lw=0.8, ls=(0, (1, 1.5)), label="Theory: ratio $=p$")
    axs[1].plot(ratio.index, ratio.values, color=BLUE, marker="o", label="Learned, naive / IPW")
    axs[1].set_xlabel("Penetration rate $p$")
    axs[1].set_ylabel("Temperature ratio")
    axs[1].set_xlim(0.05, 0.45)
    axs[1].set_ylim(0, 0.5)
    axs[1].legend(loc="upper left")
    panel_label(axs[1], 1, "Automatic tuning")
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig_temperature")


def fig_delay_gap():
    s = pd.read_csv(os.path.join(RES, "summary.csv"))
    order = ["proposed", "mp_loop", "dqn_pressure", "stratified", "cv_only", "mp_cv"]
    fig, axs = plt.subplots(2, 3, figsize=(TW, 3.2), sharex=True)
    off = dict(zip(order, np.linspace(-0.012, 0.012, len(order))))
    for j, (net, name) in enumerate(NETS):
        g = main_rows(s, net)
        for i, (metric, ylab) in enumerate((("delay_all", "Mean delay (s)"), ("delay_gap", "Delay gap (s)"))):
            ax = axs[i, j]
            for c in order:
                m, r = KEY[c]
                h = g[(g.method == m) & (g.reward == r) & g.eval_p.isin(P_LEVELS)].sort_values("eval_p")
                if h.empty:
                    continue
                st = STYLE[c]
                ax.errorbar(h.eval_p + off[c], h[metric], yerr=h[metric + "_ci"].fillna(0), color=st["color"],
                            marker=st["marker"], ls=st["ls"], label=st["label"], capsize=1.8, elinewidth=0.7,
                            markersize=3.8, lw=1.1)
            ax.set_xticks(P_LEVELS, ["0.1", "0.2", "0.4"])
            if metric == "delay_gap":
                ax.axhline(0, color=INK2, lw=0.6)
                ax.set_xlabel("Penetration rate $p$")
            if j == 0:
                ax.set_ylabel(ylab)
            panel_label(ax, 3 * i + j, name)
    axs[0, 1].set_ylim(0, 80)
    axs[0, 2].set_ylim(10, 60)
    axs[1, 0].set_ylim(-10, 90)
    axs[1, 1].set_ylim(-5, 25)
    axs[1, 2].set_ylim(-3, 22)
    h, lab = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.07))
    fig.tight_layout(h_pad=0.6, w_pad=1.0)
    save(fig, "fig_delay_gap")


def fig_fairness():
    s = pd.read_csv(os.path.join(RES, "summary.csv"))
    rewards = [("privileged", "Privil.$^\\dagger$"), ("naive", "Naive"), ("ipw", "IPW"), ("stratified", "Strat."),
               ("fused_cs", "Fused*")]
    fig, axs = plt.subplots(1, 3, figsize=(TW, 1.95), sharey=False)
    w = 0.38
    for j, (net, name) in enumerate(NETS):
        g = main_rows(s, net)
        g = g[np.isclose(g.eval_p, 0.1)]
        ax = axs[j]
        for k, (state, col, lab) in enumerate((("sac_auto", GREY2, "CV-only state"), ("sac_auto_loopstate", BLUE, "Loop--CV state"))):
            for x, (r, _) in enumerate(rewards):
                h = g[(g.method == state) & (g.reward == r)]
                if h.empty:
                    continue
                v, c = h.delay_gap.iloc[0], h.delay_gap_ci.iloc[0]
                ax.bar(x + (k - 0.5) * w, v, w * 0.92, color=col, label=lab if x == 3 else None,
                       yerr=c, error_kw=dict(elinewidth=0.6, capsize=1.5, ecolor=INK2))
        ax.set_xticks(range(len(rewards)), [n for _, n in rewards], fontsize=6.5, rotation=30, ha="right",
                      rotation_mode="anchor")
        ax.axhline(0, color=INK2, lw=0.6)
        ax.grid(axis="x", visible=False)
        if j == 0:
            ax.set_ylabel("Delay gap at $p=0.1$ (s)")
        panel_label(ax, j, name)
    h, lab = axs[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig_fairness")


def fig_drift():
    ps = pd.read_csv(os.path.join(RES, "per_seed.csv"))
    ps = ps[ps.scenario == "main"]
    tests = P_LEVELS + (1.0,)
    panels = [("1tls_3x3", "sac_auto", "stratified", "Synthetic,\nCV-only state"),
              ("1tls_3x3", "sac_auto_loopstate", "fused_cs", "Synthetic,\nproposed"),
              ("cologne1", "sac_auto_loopstate", "fused_cs", "Cologne,\nproposed"),
              ("ingolstadt1", "sac_auto_loopstate", "fused_cs", "Ingolstadt,\nproposed")]
    fig, axs = plt.subplots(1, 4, figsize=(TW, 1.75))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("blue", ["#f4f8fd", "#86b6ef", "#256abf", "#0d366b"])
    for k, (net, m, r, title) in enumerate(panels):
        ax = axs[k]
        M = np.full((3, 4), np.nan)
        med = np.full((3, 4), np.nan)
        for a, tp in enumerate(P_LEVELS):
            for b, ep in enumerate(tests):
                h = ps[(ps.net == net) & (ps.method == m) & (ps.reward == r) & np.isclose(ps.train_p, tp)
                       & np.isclose(ps.eval_p, ep)]
                if len(h):
                    M[a, b], med[a, b] = h.delay_all.mean(), h.delay_all.median()
        # colour = delay relative to the best cell of the panel (a common scale across networks)
        rel = M / np.nanmin(M)
        ax.imshow(np.minimum(rel, 2.0), cmap=cmap, vmin=1.0, vmax=2.0, aspect="auto")
        for a in range(3):
            for b in range(4):
                txt = f"{M[a, b]:.0f}" + (f"\n({med[a, b]:.0f})" if abs(M[a, b] - med[a, b]) > 0.1 * med[a, b] else "")
                ax.text(b, a, txt, ha="center", va="center", fontsize=6.2,
                        color="white" if rel[a, b] > 1.45 else INK)
        ax.set_xticks(range(4), ["0.1", "0.2", "0.4", "1.0"])
        ax.set_yticks(range(3), ["0.1", "0.2", "0.4"] if k == 0 else [])
        ax.set_xlabel("Test $p$")
        if k == 0:
            ax.set_ylabel("Training $p$")
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(f"({'abcd'[k]}) {title}", fontsize=7)
    sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=matplotlib.colors.Normalize(1, 2))
    cb = fig.colorbar(sm, ax=axs, fraction=0.025, pad=0.02)
    cb.set_label("Delay / best cell", fontsize=7)
    cb.set_ticks([1, 1.5, 2], labels=["1", "1.5", "$\\geq$2"])
    cb.outline.set_visible(False)
    save(fig, "fig_drift")


def fig_robustness():
    ps = pd.read_csv(os.path.join(RES, "per_seed.csv"))
    ps = ps[np.isclose(ps.eval_p, 0.1) & (ps.train_p.isna() | np.isclose(ps.train_p.fillna(-1), 0.1))]
    det = [("main", "None"), ("noise0.05", "5%"), ("noise0.1", "10%"), ("noise0.2", "20%"), ("fail1", "Fail")]
    dem = [("main", "$\\times$1.0"), ("scale1.3", "$\\times$1.3"), ("scale1.6", "$\\times$1.6")]
    ctrl = ["proposed", "mp_loop", "dqn_pressure", "stratified"]
    fig = plt.figure(figsize=(TW, 3.15))
    gs = fig.add_gridspec(2, 6)
    axes = [fig.add_subplot(gs[0, 0:2]), fig.add_subplot(gs[0, 2:4]), fig.add_subplot(gs[0, 4:6]),
            fig.add_subplot(gs[1, 1:3]), fig.add_subplot(gs[1, 3:5])]
    specs = [(net, name, det) for net, name in NETS] + [(net, name, dem) for net, name in NETS[1:]]
    off = dict(zip(ctrl, np.linspace(-0.24, 0.24, len(ctrl))))
    for k, (ax, (net, name, conds)) in enumerate(zip(axes, specs)):
        for c in ctrl:
            m, r = KEY[c]
            st = STYLE[c]
            means, x = [], []
            for i, (sc, _) in enumerate(conds):
                h = ps[(ps.net == net) & (ps.method == m) & (ps.reward == r) & (ps.scenario == sc)]
                if h.empty:
                    continue
                ax.scatter(np.full(len(h), i + off[c]), h.delay_all, s=5, color=st["color"], alpha=0.45, lw=0,
                           marker=st["marker"])
                means.append(h.delay_all.mean())
                x.append(i + off[c])
            ax.plot(x, means, color=st["color"], marker=st["marker"], ls="none", markersize=4.5,
                    markeredgecolor="white", markeredgewidth=0.5, label=st["label"])
        ax.set_xticks(range(len(conds)), [n for _, n in conds], fontsize=6.8)
        ax.grid(axis="x", visible=False)
        if conds is det and k == 1:
            ax.set_xlabel("Relative s.d. of loop-count noise, or one failed loop", fontsize=7)
        else:
            ax.set_yscale("log")
            ax.set_yticks([10, 20, 50, 100, 200], ["10", "20", "50", "100", "200"])
            ax.minorticks_off()
            ax.set_xlabel("Demand scale", fontsize=7)
            if net == "ingolstadt1":
                ax.set_ylim(12, 80)
        if k in (0, 3):
            ax.set_ylabel("Mean delay (s)")
        panel_label(ax, k, f"{name}, {'detector errors' if conds is det else 'heavier demand'}")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout(h_pad=0.8, w_pad=0.8)
    save(fig, "fig_robustness")


def fig_networks_seeds():
    """Multi-signal and held-out networks: every seed as a dot, the mean as a bar, max-pressure as a line."""
    ps = pd.read_csv(os.path.join(RES, "per_seed.csv"))
    ps = ps[ps.scenario == "main"]
    groups = [("cologne3", 0.1, "Cologne, 3 signals\n$p=0.1$"), ("cologne3", 0.2, "Cologne, 3 signals\n$p=0.2$"),
              ("ingolstadt7", 0.1, "Ingolstadt, 7 signals\n(held out), $p=0.1$"),
              ("cologne8", 0.1, "Cologne, 8 signals\n(held out), $p=0.1$")]
    ctrl = [("proposed", ("isac_auto_loopstate", "fused_cs")), ("stratified", ("isac_auto_loopstate", "stratified")),
            ("dqn_pressure", ("idqn_loopstate", "pressure"))]
    fig, axs = plt.subplots(1, 4, figsize=(TW, 1.85))
    for k, (net, p, title) in enumerate(groups):
        ax = axs[k]
        for i, (c, (m, r)) in enumerate(ctrl):
            h = ps[(ps.net == net) & (ps.method == m) & (ps.reward == r) & np.isclose(ps.eval_p, p)
                   & np.isclose(ps.train_p, p)]
            st = STYLE[c]
            if h.empty:
                continue
            ax.bar(i, h.delay_all.mean(), 0.62, color=st["color"], alpha=0.28, label=st["label"])
            ax.scatter(np.full(len(h), i), h.delay_all, color=st["color"], marker=st["marker"], s=14, zorder=3,
                       edgecolor="white", lw=0.4)
        mp = ps[(ps.net == net) & (ps.method == "max_pressure_loop") & np.isclose(ps.eval_p, p)].delay_all
        if len(mp):
            ax.axhline(mp.iloc[0], color=ORANGE, lw=1.2, label=STYLE["mp_loop"]["label"])
        ax.set_xticks([])
        ax.grid(axis="x", visible=False)
        ax.set_ylim(0, max(ax.get_ylim()[1], 50))
        if k == 0:
            ax.set_ylabel("Network delay (s)")
        ax.set_title(f"({'abcd'[k]}) {title}", fontsize=7)
    h, lab = axs[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout(w_pad=0.8)
    save(fig, "fig_multinet")


def fig_learning_curves():
    pats = {"proposed": ["runs/fused_cs/{n}/fused_cs_p0.1_s*"],
            "stratified": ["runs/real/{n}/sac_auto_loopstate/stratified_p0.1_s*", "runs/sac_auto_loopstate/stratified_p0.1_s*"],
            "dqn_pressure": ["runs/pressure/{n}/dqn_loopstate/pressure_p0.1_s*"],
            "cv_only": ["runs/real/{n}/sac_auto/stratified_p0.1_s*", "runs/sac_auto/stratified_p0.1_s*"]}
    fig, axs = plt.subplots(1, 3, figsize=(TW, 1.95))
    for j, (net, name) in enumerate(NETS):
        ax = axs[j]
        n = "synthetic" if net == "1tls_3x3" else net
        for c in ("cv_only", "stratified", "dqn_pressure", "proposed"):
            pat = pats[c][0] if (net != "1tls_3x3" or len(pats[c]) == 1) else pats[c][1]
            fs = sorted(glob.glob(os.path.join(ROOT, pat.format(n=n), "train_log.csv")))
            curves = []
            for f in fs:
                log = pd.read_csv(f)
                log["bin"] = (log.step // 10_000) * 10_000
                curves.append(log[log.step < 170_000].groupby("bin").true_cost_mean.mean())
            if not curves:
                continue
            C = pd.concat(curves, axis=1)
            mu, sd = C.mean(axis=1), C.std(axis=1)
            st = STYLE[c]
            ax.plot(C.index / 1000, mu, color=st["color"], ls=st["ls"], lw=1.1, label=st["label"])
            ax.fill_between(C.index / 1000, mu - sd, mu + sd, color=st["color"], alpha=0.13, lw=0)
        ax.set_xlabel("Training decisions ($10^3$)")
        if j == 0:
            ax.set_ylabel("True cost per decision")
        panel_label(ax, j, f"{name}, $p=0.1$")
    h, lab = axs[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig_learning")


def fig_network_maps():
    nets = [("1tls_3x3", "Synthetic\n1 signal"), ("cologne1", "Cologne\n1 signal"), ("ingolstadt1", "Ingolstadt\n1 signal"),
            ("cologne3", "Cologne\n3 signals"), ("ingolstadt7", "Ingolstadt, 7\n(held out)"), ("cologne8", "Cologne, 8\n(held out)")]
    fig, axs = plt.subplots(1, 6, figsize=(TW, 1.45))
    for k, (net, title) in enumerate(nets):
        ax = axs[k]
        f = glob.glob(os.path.join(ROOT, "tsc", "net", net, "*.net.xml"))[0]
        root = ET.parse(f).getroot()
        for e in root.iter("edge"):
            if e.get("function") == "internal":
                continue
            for ln in e.iter("lane"):
                xy = np.array([list(map(float, pt.split(","))) for pt in ln.get("shape").split()])
                ax.plot(xy[:, 0], xy[:, 1], color=GREY2, lw=0.35, solid_capstyle="round")
        tls = [j for j in root.iter("junction") if j.get("type", "").startswith("traffic_light")]
        pts = np.array([[float(j.get("x")), float(j.get("y"))] for j in tls])
        if len(pts):
            ax.scatter(pts[:, 0], pts[:, 1], s=9, color=BLUE, zorder=3, edgecolor="white", lw=0.4)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"({'abcdef'[k]}) {title}", fontsize=6.5, loc="center")
    fig.tight_layout(w_pad=0.3)
    save(fig, "fig_networks")


def fig_diagnosis():
    """Held-out Ingolstadt network, minor entry road (results/diagnostics/lanediag_fig.py, one shared scenario)."""
    fs = sorted(glob.glob(os.path.join(RES, "diagnostics", "lanediag", "*.csv")))
    if not fs:
        return
    d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    d["reward"] = d.model.str.replace(r"_p0\.1_s\d", "", regex=True)
    rewards = [("stratified", "Lane-\nstratified"), ("fused", "Fused,\npooled"), ("fused_cs", "Fused\n(proposed)")]
    fig, axs = plt.subplots(1, 2, figsize=(TW * 0.82, 1.9))
    w = 0.36
    for x, (r, _) in enumerate(rewards):
        g = d[d.reward == r]
        per = g.groupby("model")
        nocv = per.apply(lambda h: (h.n == 0).mean())
        # (a) estimate / true cost on decisions without a CV in the lane, for the fallback the model was trained with
        col = "est_same" if r == "fused_cs" else "est_pooled"
        ratio = per.apply(lambda h: h[h.n == 0][col].sum() / h[h.n == 0].true.sum())
        axs[0].bar(x, ratio.mean(), 0.55, color=GREY2)
        axs[0].scatter(np.full(len(ratio), x), ratio, s=8, color=INK2, zorder=3)
        # (b) share of decisions that give the lane green, without and with a CV in it
        for k, (mask, lab, colr) in enumerate(((lambda h: h.n == 0, "no CV in the lane", GREY2),
                                              (lambda h: h.n > 0, "at least one CV", BLUE))):
            sh = per.apply(lambda h: h[mask(h)].served.mean())
            axs[1].bar(x + (k - 0.5) * w, 100 * sh.mean(), w * 0.92, color=colr, label=lab if x == 0 else None)
            axs[1].scatter(np.full(len(sh), x + (k - 0.5) * w), 100 * sh, s=7, color=INK2, zorder=3)
    for ax in axs:
        ax.set_xticks(range(len(rewards)), [n for _, n in rewards], fontsize=6.5)
        ax.grid(axis="x", visible=False)
    axs[0].axhline(1, color=INK2, lw=0.7, ls=(0, (1, 1.5)))
    axs[0].set_ylim(0, 1.1)
    axs[0].set_ylabel("Estimate / true cost")
    panel_label(axs[0], 0, "Lane cost without a CV")
    axs[1].set_ylabel("Decisions with green (%)")
    axs[1].set_ylim(0, 45)
    h, lab = axs[1].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=2, bbox_to_anchor=(0.62, -0.1))
    panel_label(axs[1], 1, "Service of the minor road")
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig_diagnosis")


def main():
    fig_network_maps()
    fig_estimators()
    fig_temperature()
    fig_delay_gap()
    fig_fairness()
    fig_drift()
    fig_robustness()
    fig_networks_seeds()
    fig_learning_curves()
    fig_diagnosis()
    print("figures written to", OUT)


if __name__ == "__main__":
    main()
