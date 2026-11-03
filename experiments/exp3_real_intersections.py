"""Stage 3: loop-CV state on the synthetic and real single intersections (SAC + loop-count state).

Runs in this order (resumable; finished outputs are skipped):

  A. Final model on the synthetic intersection, completing the reward comparison
       loop-state SAC x naive/ipw/loop_count x p 0.1/0.2/0.4 x 5 seeds   (privileged/stratified already done)
       loop-state SAC x privileged at p = 1.0 x 5 seeds                    (full-detection reference)
       then their evaluations: cross-p (0.1/0.2/0.4/1.0) and fixed/peak demand
  B. Real intersections (RESCO cologne1, ingolstadt1)
       final model:  loop-state SAC x 5 rewards x p 0.1/0.2/0.4 x 5 seeds, plus privileged at p = 1.0
       ablation:     CV-only-state SAC x privileged/stratified x p 0.1/0.2/0.4 x 3 seeds
       base paper:   QM-ATSC (adapted) x p 0.1/0.2/0.4/1.0 x 5 seeds
       baselines:    fixed-time, max-pressure (CV), max-pressure (loops) x p 0.1/0.2/0.4/1.0
  C. Sensitivity of the final model (synthetic, stratified reward, p = 0.2, 3 seeds)
       target entropy ratio 0.3 / 0.7 (main 0.5); reward scale K = 25 / 100 (main 50)

    python experiments/exp3_real_intersections.py --gpu-workers 3 --cpu-workers 8
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess

from exp2_ablations import PY, ROOT, _env, run_pool

NETS = ("cologne1", "ingolstadt1")
P_LEVELS = (0.1, 0.2, 0.4)
REWARDS = ("privileged", "naive", "ipw", "stratified", "loop_count")
BASELINES = ("fixed_time", "max_pressure_cv", "max_pressure_loop")


def stage_a():
    base = os.path.join(ROOT, "runs", "sac_auto_loopstate")
    js = [("rl", os.path.join(base, f"{r}_p{p}_s{s}"), ["--obs-loop", "--reward", r, "--p", str(p), "--seed", str(s)], p)
          for r, p, s in itertools.product(("naive", "ipw", "loop_count"), P_LEVELS, range(5))]
    js += [("rl", os.path.join(base, f"privileged_p1.0_s{s}"), ["--obs-loop", "--reward", "privileged", "--p", "1.0",
                                                                 "--seed", str(s)], 1.0) for s in range(5)]
    return js


def stage_a_evals():
    """Cross-p and demand evaluations for every finished synthetic loop-state run."""
    out = []
    base = os.path.join(ROOT, "runs", "sac_auto_loopstate")
    for d in sorted(os.listdir(base)):
        d = os.path.join(base, d)
        if not os.path.exists(os.path.join(d, "DONE")):
            continue
        p = json.load(open(os.path.join(d, "config.json")))["p"]
        for q in P_LEVELS + (1.0,):
            if q != p:
                out.append(("eval", os.path.join(d, f"eval_agent_p{q}.csv"), ["--run", d, "--p", str(q)], q))
        for dem in ("fixed", "peak"):
            out.append(("eval", os.path.join(d, f"eval_agent_p{p}_demand_{dem}.csv"),
                        ["--run", d, "--p", str(p), "--demand", dem], p))
    return [j for j in out if not os.path.exists(j[1])]


def stage_b():
    js = []
    for net in NETS:
        base = os.path.join(ROOT, "runs", "real", net)
        for b, p in itertools.product(BASELINES, P_LEVELS + (1.0,)):
            js.append(("base", os.path.join(base, "baselines"), [b, net], p))
        for p, s in itertools.product(P_LEVELS + (1.0,), range(5)):
            js.append(("fu", os.path.join(base, "fu2026", f"p{p}_s{s}"), ["--net", net, "--p", str(p), "--seed", str(s)], p))
        for r, p, s in itertools.product(REWARDS, P_LEVELS, range(5)):
            js.append(("rl", os.path.join(base, "sac_auto_loopstate", f"{r}_p{p}_s{s}"),
                       ["--net", net, "--obs-loop", "--reward", r, "--p", str(p), "--seed", str(s)], p))
        for s in range(5):
            js.append(("rl", os.path.join(base, "sac_auto_loopstate", f"privileged_p1.0_s{s}"),
                       ["--net", net, "--obs-loop", "--reward", "privileged", "--p", "1.0", "--seed", str(s)], 1.0))
        for r, p, s in itertools.product(("privileged", "stratified"), P_LEVELS, range(3)):
            js.append(("rl", os.path.join(base, "sac_auto", f"{r}_p{p}_s{s}"),
                       ["--net", net, "--reward", r, "--p", str(p), "--seed", str(s)], p))
    return js


def stage_c():
    sens = os.path.join(ROOT, "runs", "sensitivity")
    return [("rl", os.path.join(sens, tag, f"stratified_p0.2_s{s}"),
             ["--obs-loop", flag, val, "--reward", "stratified", "--p", "0.2", "--seed", str(s)], 0.2)
            for (flag, val, tag), s in itertools.product((("--target-entropy-ratio", "0.3", "H0.3"),
                                                          ("--target-entropy-ratio", "0.7", "H0.7"),
                                                          ("--reward-scale", "25", "K25"),
                                                          ("--reward-scale", "100", "K100")), range(3))]


def run(job, device):
    kind, out, args, p = job
    if kind == "eval":
        subprocess.run([PY, os.path.join(ROOT, "scripts", "evaluate.py"), *args], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, env=_env("cpu"))
        return out
    os.makedirs(out, exist_ok=True)
    if kind == "base":
        name, net = args
        if not os.path.exists(os.path.join(out, f"eval_{name}_p{p}.csv")):
            subprocess.run([PY, os.path.join(ROOT, "scripts", "evaluate.py"), "--baseline", name, "--net", net,
                            "--p", str(p), "--out", out], check=True, stdout=subprocess.DEVNULL, env=_env("cpu"))
        return f"{name} {net} p={p}"
    with open(os.path.join(out, "stdout.log"), "a") as f:
        if kind == "fu":
            if not os.path.exists(os.path.join(out, "DONE")):
                subprocess.run([PY, os.path.join(ROOT, "scripts", "train_fu2026.py"), *args, "--out", out],
                               stdout=f, stderr=subprocess.STDOUT, check=True, env=_env("cpu"))
            if not os.path.exists(os.path.join(out, f"eval_fu2026_p{p}.csv")):
                subprocess.run([PY, os.path.join(ROOT, "scripts", "evaluate.py"), "--baseline", "fu2026",
                                "--qtable", os.path.join(out, "qtable.npy"), "--p", str(p), "--out", out],
                               stdout=f, stderr=subprocess.STDOUT, check=True, env=_env("cpu"))
            return out
        if not os.path.exists(os.path.join(out, "DONE")):
            subprocess.run([PY, os.path.join(ROOT, "scripts", "train.py"), *args, "--out", out, "--device", device],
                           stdout=f, stderr=subprocess.STDOUT, check=True, env=_env(device))
        if not os.path.exists(os.path.join(out, f"eval_agent_p{p}.csv")):
            subprocess.run([PY, os.path.join(ROOT, "scripts", "evaluate.py"), "--run", out, "--p", str(p)],
                           stdout=f, stderr=subprocess.STDOUT, check=True, env=_env("cpu"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu-workers", type=int, default=3)
    ap.add_argument("--cpu-workers", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    a, b, c = stage_a(), stage_b(), stage_c()
    n_rl = sum(j[0] == "rl" for j in a + b + c)
    print(f"A: {len(a)} runs | B: {len(b)} jobs | C: {len(c)} runs | {n_rl} RL training runs in total", flush=True)
    if args.dry_run:
        return
    run_pool(a, run, args.gpu_workers, args.cpu_workers, "A-train")
    run_pool(stage_a_evals(), run, 0, args.gpu_workers + args.cpu_workers + 3, "A-eval")
    print("STAGE A DONE", flush=True)
    b.sort(key=lambda j: {"base": 0, "fu": 1, "rl": 2}[j[0]])  # cheap jobs first
    run_pool(b, run, args.gpu_workers, args.cpu_workers, "B")
    print("STAGE B DONE", flush=True)
    run_pool(c, run, args.gpu_workers, args.cpu_workers, "C")
    print("stage 3 done", flush=True)


if __name__ == "__main__":
    main()
