"""Aggregate all evaluation results into tables and figures.

Unit of analysis = one training seed (its metrics averaged over the 20
evaluation episodes). Intervals are 95% t-intervals across seeds; reward
comparisons use Welch's t-test across seeds.

Output (results/):
  per_seed.csv            one row per (method, reward, train_p, eval_p, scenario, seed)
  summary.csv / .md       all groups, mean +- CI
  welch_tests.csv         stratified and fused_cs vs every other reward, per net, method and p
  fig_delay_vs_p.png, fig_gap_vs_p.png

    python analysis/analyze.py
"""

from __future__ import annotations

import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results")
METRICS = ["delay_all", "delay_cv", "delay_noncv", "delay_gap", "waiting_all", "unfinished", "switch_rate"]
REWARD_ORDER = ["privileged", "naive", "ipw", "stratified", "loop_count"]
KEYS = ["net", "method", "reward", "train_p", "eval_p", "scenario"]


def load():
    frames = []
    for f in glob.glob(os.path.join(ROOT, "runs", "**", "eval_*.csv"), recursive=True):
        d = pd.read_csv(f)
        if "scenario" not in d:
            d["scenario"] = "main"
        if "net" not in d:
            d["net"] = "1tls_3x3"
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df["scenario"] = df["scenario"].fillna("main")
    df["eval_p"] = df["eval_p"].astype(float)
    df["train_p"] = pd.to_numeric(df["train_p"], errors="coerce")  # "-" for baselines
    df["seed"] = df["seed"].astype(str)
    return df


def ci(x):
    x = np.asarray(x, float)
    if len(x) < 2:
        return float(np.mean(x)), float("nan")
    return float(x.mean()), float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))


def summarise(per_seed):
    rows = []
    for key, g in per_seed.groupby(KEYS, dropna=False):
        row = dict(zip(KEYS, key), n_seeds=len(g))
        for k in METRICS:
            row[k], row[k + "_ci"] = ci(g[k])
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["net", "scenario", "eval_p", "method", "reward"])


def main_view(s, net="1tls_3x3"):
    """Main study: default demand, evaluated at the training p (baselines at every p)."""
    same_p = s.train_p.isna() | np.isclose(s.train_p.fillna(-1), s.eval_p)
    return s[(s.scenario == "main") & same_p & (s.net == net)]


def write_md(summary, path):
    syn = summary[summary.net == "1tls_3x3"]
    sections = {
        "Main study, synthetic intersection (train p = test p, random demand)": main_view(summary),
        "Cross-p transfer (SAC, auto temperature)": syn[(syn.scenario == "main") & (syn.method == "sac_auto")
                                                        & ~np.isclose(syn.train_p.fillna(-1), syn.eval_p)],
    }
    for sc in sorted(set(syn.scenario) - {"main"}):
        sections[f"Scenario: {sc}"] = syn[syn.scenario == sc]
    for net in sorted(set(summary.net) - {"1tls_3x3"}):
        sections[f"Real intersection: {net}"] = main_view(summary, net)
    with open(path, "w") as f:
        for title, s in sections.items():
            if s.empty:
                continue
            f.write(f"\n## {title}\n\n| method | reward | train p | test p | seeds | delay (s) | CV delay | non-CV delay | gap (s) | unfinished |\n")
            f.write("|---|---|---|---|---|---|---|---|---|---|\n")
            for _, r in s.iterrows():
                tp = "-" if np.isnan(r.train_p) else r.train_p
                f.write(f"| {r.method} | {r.reward} | {tp} | {r.eval_p} | {r.n_seeds} | {r.delay_all:.1f} ± {r.delay_all_ci:.1f} | "
                        f"{r.delay_cv:.1f} | {r.delay_noncv:.1f} | {r.delay_gap:.1f} ± {r.delay_gap_ci:.1f} | {r.unfinished:.0f} |\n")


def figures(summary):
    s = main_view(summary)
    for metric, ylabel, fname in (("delay_all", "Mean delay per vehicle (s)", "fig_delay_vs_p.png"),
                                  ("delay_gap", "Non-CV minus CV delay (s)", "fig_gap_vs_p.png")):
        fig, ax = plt.subplots(figsize=(6.4, 4))
        for r in REWARD_ORDER:
            g = s[(s.method == "sac_auto") & (s.reward == r)].sort_values("eval_p")
            if len(g):
                ax.errorbar(g.eval_p, g[metric], yerr=g[metric + "_ci"].fillna(0), marker="o", capsize=3, label=f"SAC, {r}")
        for b, ls in (("max_pressure_loop", "--"), ("max_pressure_cv", ":"), ("fixed_time", "-."), ("fu2026_qm_atsc", "-")):
            g = s[s.method == b].sort_values("eval_p")
            if len(g):
                ax.plot(g.eval_p, g[metric], ls, color="grey" if b != "fu2026_qm_atsc" else "black", label=b)
        ax.set_xscale("log")
        ax.set_xticks([0.1, 0.2, 0.4, 1.0], ["10%", "20%", "40%", "100%"])
        ax.set_xlabel("CV penetration rate p")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, fname), dpi=200)
        plt.close(fig)


REFERENCES = (("sac_auto", "stratified"), ("sac_auto_loopstate", "stratified"), ("isac_auto_loopstate", "stratified"),
              ("sac_auto_loopstate", "fused_cs"), ("isac_auto_loopstate", "fused_cs"))


def paired_tests(df):
    """Scenario-paired comparison: for each p, average each method over its seeds per evaluation
    episode, then compare each reference (stratified reward with the CV-only state, loop-CV state and
    multi-intersection; final fused_cs reward with the loop-CV state) with every other method on the same 20 scenarios (paired t-test and Wilcoxon)."""
    rows = []
    d = df[(df.scenario == "main")]
    same = d.train_p.isna() | np.isclose(d.train_p.fillna(-1), d.eval_p)
    d = d[same]
    for (net, p), g in d.groupby(["net", "eval_p"]):
        per_ep = g.groupby(["method", "reward", "episode"])["delay_all"].mean()
        present = set(per_ep.index.droplevel(2))
        for ref_key in REFERENCES:
            if ref_key not in present:
                continue
            ref = per_ep.loc[ref_key]
            for (m, r) in sorted(present):
                if (m, r) == ref_key:
                    continue
                other = per_ep.loc[(m, r)]
                common = ref.index.intersection(other.index)
                if len(common) < 5:
                    continue
                a, b = ref[common], other[common]
                t, pt = stats.ttest_rel(a, b)
                try:
                    pw = stats.wilcoxon(a, b).pvalue
                except ValueError:
                    pw = float("nan")
                rows.append({"net": net, "p": p, "reference": f"{ref_key[0]}/{ref_key[1]}",
                             "comparison": f"{ref_key[0]}/{ref_key[1]} vs {m}/{r}", "n_scenarios": len(common),
                             "mean_diff_s": float((a - b).mean()), "reference_better_in": int((a < b).sum()),
                             "paired_t_p": pt, "wilcoxon_p": pw})
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "paired_tests.csv"), index=False)


def learning_curves():
    """True-cost learning curves (mean over seeds) and training stability per reward."""
    rows = []
    for f in glob.glob(os.path.join(ROOT, "runs", "sac_auto", "*", "train_log.csv")):
        name = os.path.basename(os.path.dirname(f))
        reward, p, _ = name.rsplit("_", 2)
        try:
            log = pd.read_csv(f)
        except Exception:
            continue
        log["reward"], log["p"] = reward, float(p[1:])
        rows.append(log)
    if not rows:
        return
    logs = pd.concat(rows, ignore_index=True)
    logs["bin"] = (logs["step"] // 10_000) * 10_000
    curve = logs.groupby(["p", "reward", "bin"])["true_cost_mean"].mean().reset_index()
    curve.to_csv(os.path.join(OUT, "learning_curves.csv"), index=False)
    ps = sorted(curve.p.unique())
    fig, axs = plt.subplots(1, len(ps), figsize=(4.2 * len(ps), 3.4), squeeze=False)
    for ax, p in zip(axs[0], ps):
        for r in REWARD_ORDER:
            g = curve[(curve.p == p) & (curve.reward == r)]
            if len(g):
                ax.plot(g["bin"] / 1000, g["true_cost_mean"], label=r)
        ax.set_title(f"p = {p}", fontsize=10)
        ax.set_xlabel("training steps (thousands)")
        ax.set_ylabel("true cost per decision")
        ax.grid(alpha=0.3)
    axs[0][0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_learning_curves.png"), dpi=200)
    plt.close(fig)
    # stability: critic loss in the last 20% of training, per reward and p
    late = logs[logs["step"] > 0.8 * logs["step"].max()]
    late.groupby(["p", "reward"])[["q_loss", "true_cost_mean"]].agg(["mean", "std"]).to_csv(
        os.path.join(OUT, "training_stability.csv"))


def main():
    os.makedirs(OUT, exist_ok=True)
    df = load()
    per_seed = df.groupby(KEYS + ["seed"], as_index=False, dropna=False)[METRICS].mean()
    per_seed.to_csv(os.path.join(OUT, "per_seed.csv"), index=False)
    summary = summarise(per_seed)
    summary.to_csv(os.path.join(OUT, "summary.csv"), index=False)
    write_md(summary, os.path.join(OUT, "summary.md"))

    tests = []
    main = per_seed[(per_seed.scenario == "main") & per_seed.method.isin([m for m, _ in REFERENCES])
                    & np.isclose(per_seed.train_p, per_seed.eval_p)]
    for (net, method, p), g in main.groupby(["net", "method", "eval_p"]):
        for ref_r in ("stratified", "fused_cs"):
            ref = g[g.reward == ref_r]
            for r in REWARD_ORDER + ["pressure", "fused", "fused_cs"]:
                other = g[g.reward == r]
                if r == ref_r or len(ref) < 2 or len(other) < 2:
                    continue
                for m in ("delay_all", "delay_gap"):
                    t, pv = stats.ttest_ind(ref[m], other[m], equal_var=False)
                    tests.append({"net": net, "method": method, "p": p, "metric": m, "comparison": f"{ref_r} vs {r}",
                                  "diff": ref[m].mean() - other[m].mean(), "t": t, "p_value": pv})
    pd.DataFrame(tests).to_csv(os.path.join(OUT, "welch_tests.csv"), index=False)
    paired_tests(df)
    learning_curves()
    figures(summary)
    print(main_view(summary)[["method", "reward", "eval_p", "n_seeds", "delay_all", "delay_all_ci", "delay_gap"]]
          .round(2).to_string(index=False))


if __name__ == "__main__":
    main()
