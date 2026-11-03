"""Stage 7: complete the proposed model (fused_cs) and test its robustness.

  1. fused_cs x p 0.2/0.4 x cologne1, ingolstadt1, synthetic x 5 seeds (p 0.1 was in stage 6)
  2. fused_cs x cologne3 x p 0.2 x 3 seeds
  3. penetration drift: every fused_cs single-intersection model tested at the other p (0.1/0.2/0.4/1.0)
  4. unseen demand (fixed, peak hour): synthetic fused_cs models at their training p
  5. loop-detector imperfections at test time (p = 0.1): count noise 5/10/20 % and one failed loop,
     for fused_cs, stratified (loop-CV state) and PressLight-style DQN, and max-pressure with loops
  6. heavier real demand (p = 0.1): cologne1 and ingolstadt1 demand x1.3 and x1.6, for the same
     learners plus SAC with pressure, and the baselines

Training first (resumable), then all evaluations; finished outputs are skipped.

    python experiments/exp7_final_model_tests.py --gpu-workers 3 --cpu-workers 8
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess

from exp2_ablations import PY, ROOT, _env, run_pool
from exp6_same_signal_fallback import BUDGET

LAM = json.load(open(os.path.join(ROOT, "runs", "fused_tuning", "CHOSEN.json")))["lambda"]
NETS = ("cologne1", "ingolstadt1", "1tls_3x3")
RAM_GB = {"single": 3.0, "multi": 4.0, "eval": 1.0}


def fused_dir(net, p, s):
    return os.path.join(ROOT, "runs", "fused_cs", "synthetic" if net == "1tls_3x3" else net, f"fused_cs_p{p}_s{s}")


def base_dir(net):
    return os.path.join(ROOT, "runs", "baselines") if net == "1tls_3x3" else os.path.join(ROOT, "runs", "real", net, "baselines")


def learner_dirs(net, p, with_sac_pressure=False):
    """Trained runs (directories) of the learners compared under test-time conditions."""
    real = net != "1tls_3x3"
    loop = os.path.join(ROOT, "runs", "real", net, "sac_auto_loopstate") if real else os.path.join(ROOT, "runs", "sac_auto_loopstate")
    pr = os.path.join(ROOT, "runs", "pressure", net if real else "synthetic")
    ds = [fused_dir(net, p, s) for s in range(5)]
    ds += [os.path.join(loop, f"stratified_p{p}_s{s}") for s in range(5)]
    ds += [os.path.join(pr, "dqn_loopstate", f"pressure_p{p}_s{s}") for s in range(5)]
    if with_sac_pressure:
        ds += [os.path.join(pr, "sac_auto_loopstate", f"pressure_p{p}_s{s}") for s in range(5)]
    return ds


def train_jobs():
    js = [("single", fused_dir(net, p, s),
           ["--net", net, "--obs-loop", "--reward", "fused_cs", "--fused-lambda", str(LAM), "--p", str(p), "--seed", str(s)])
          for p in (0.2, 0.4) for net in NETS for s in range(5)]
    js += [("multi", os.path.join(ROOT, "runs", "multi", "cologne3", "isac_auto_loopstate", f"fused_cs_p0.2_s{s}"),
            ["--net", "cologne3", "--obs-loop", "--reward", "fused_cs", "--fused-lambda", str(LAM), "--p", "0.2", "--seed", str(s)])
           for s in range(5)]
    return js


def eval_jobs():
    js = []
    # 3. penetration drift
    for net in NETS:
        for p in (0.1, 0.2, 0.4):
            for s in range(5):
                for q in (0.1, 0.2, 0.4, 1.0):
                    if q != p:
                        js.append(("eval", ["--run", fused_dir(net, p, s), "--p", str(q)]))
    # 4. unseen demand
    for p in (0.1, 0.2, 0.4):
        for s in range(5):
            for dem in ("fixed", "peak"):
                js.append(("eval", ["--run", fused_dir("1tls_3x3", p, s), "--p", str(p), "--demand", dem]))
    # 5. loop-detector imperfections
    imperfect = [["--loop-noise", "0.05"], ["--loop-noise", "0.1"], ["--loop-noise", "0.2"], ["--loop-fail"]]
    for net in NETS:
        for cond in imperfect:
            js += [("eval", ["--run", d, "--p", "0.1", *cond]) for d in learner_dirs(net, 0.1)]
            js.append(("eval", ["--baseline", "max_pressure_loop", "--net", net, "--p", "0.1", "--out", base_dir(net), *cond]))
    # 6. heavier real demand
    for net in ("cologne1", "ingolstadt1"):
        for sc in ("1.3", "1.6"):
            js += [("eval", ["--run", d, "--p", "0.1", "--demand-scale", sc]) for d in learner_dirs(net, 0.1, True)]
            js += [("eval", ["--baseline", b, "--net", net, "--p", "0.1", "--out", base_dir(net), "--demand-scale", sc])
                   for b in ("fixed_time", "max_pressure_cv", "max_pressure_loop")]
    return js


def _expected_eval_file(args):
    """Path evaluate.py will write, so finished evaluations are skipped on resume."""
    a = dict(zip(args[::2], args[1::2])) if "--loop-fail" not in args else dict(zip([x for x in args if x != "--loop-fail"][::2],
                                                                                    [x for x in args if x != "--loop-fail"][1::2]))
    parts = [f"demand_{a['--demand']}" if a.get("--demand", "random") != "random" else "",
             f"scale{float(a['--demand-scale']):g}" if "--demand-scale" in a else "",
             f"noise{float(a['--loop-noise']):g}" if "--loop-noise" in a else "",
             "fail1" if "--loop-fail" in args else ""]
    scen = "-".join(x for x in parts if x)
    tag = f"p{a['--p']}" + (f"_{scen}" if scen else "")
    out = a.get("--out") or a["--run"]
    return os.path.join(out, f"eval_{a.get('--baseline', 'agent')}_{tag}.csv")


def run(job, device):
    kind = job[0]
    BUDGET.take(RAM_GB[kind])
    try:
        if kind == "eval":
            args = job[1]
            if not os.path.exists(_expected_eval_file(args)):
                subprocess.run([PY, os.path.join(ROOT, "scripts", "evaluate.py"), *args], check=True,
                               stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, env=_env("cpu"))
            return " ".join(args[:4])
        _, out, args = job
        os.makedirs(out, exist_ok=True)
        train, ev = ("train_multi.py", "evaluate_multi.py") if kind == "multi" else ("train.py", "evaluate.py")
        p = args[args.index("--p") + 1]
        with open(os.path.join(out, "stdout.log"), "a") as f:
            if not os.path.exists(os.path.join(out, "DONE")):
                subprocess.run([PY, os.path.join(ROOT, "scripts", train), *args, "--out", out, "--device", device],
                               stdout=f, stderr=subprocess.STDOUT, check=True, env=_env(device))
            if not os.path.exists(os.path.join(out, f"eval_agent_p{p}.csv")):
                subprocess.run([PY, os.path.join(ROOT, "scripts", ev), "--run", out, "--p", p],
                               stdout=f, stderr=subprocess.STDOUT, check=True, env=_env("cpu"))
        return out
    finally:
        BUDGET.give(RAM_GB[kind])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu-workers", type=int, default=3)
    ap.add_argument("--cpu-workers", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    t, e = train_jobs(), eval_jobs()
    print(f"stage 7: lambda = {LAM}; {len(t)} training runs, then {len(e)} evaluations", flush=True)
    if args.dry_run:
        missing = [d for d in {a[1] for _, a in [(j[0], j[1]) for j in e] if a[0] == "--run"}
                   if not os.path.exists(os.path.join(d, "model.pt")) and "fused_cs_p0.1" not in d and "fused_cs" not in d]
        print("missing trained runs (excluding those trained here):", missing)
        return
    run_pool(t, run, args.gpu_workers, args.cpu_workers, "X6-train")
    print("TRAINING DONE", flush=True)
    run_pool(e, run, 0, args.gpu_workers + args.cpu_workers, "X6-eval")
    print("stage 7 done", flush=True)


if __name__ == "__main__":
    main()
