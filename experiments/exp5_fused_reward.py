"""Stage 5: the fused loop-CV reward, tuned on validation episodes, then tested everywhere.

Final model: SAC + loop-CV state + fused reward r = -(C_str + lambda * P) / K
(stratified cost plus pressure as a discharge term). Protocol, fixed before any run:

  T. Tuning, synthetic intersection only, p = 0.2, 3 seeds per lambda in {0.5, 1, 2}.
     Each model is scored on 10 validation episodes (seeds 8e8 + k, disjoint from training
     and from the 20 test scenarios); lambda* = lowest mean validation delay (ties: smaller
     lambda). Tuning runs are never evaluated on the test scenarios. Saved to
     runs/fused_tuning/CHOSEN.json and reused on resume.
  M. Main runs with lambda*, all other hyperparameters unchanged:
       cologne1, ingolstadt1, synthetic x p 0.1/0.2/0.4 x 5 seeds; cologne3 x p 0.1/0.2 x 3 seeds
  H. Held-out network never used during design: RESCO ingolstadt7 (7 signals), p = 0.1
       baselines (fixed time, max-pressure CV / loops), fused SAC, PressLight-style DQN and
       stratified SAC, 3 seeds each

    python experiments/exp5_fused_reward.py --gpu-workers 3 --cpu-workers 8
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess

import pandas as pd

from exp2_ablations import PY, ROOT, _env, run_pool
from exp4_multisignal_pressure import BUDGET

LAMBDAS = (0.5, 1.0, 2.0)
TUNE_P = 0.2
RAM_GB = {"base": 2.0, "single": 3.0, "multi": 4.0, "multi7": 9.0}
CHOSEN = os.path.join(ROOT, "runs", "fused_tuning", "CHOSEN.json")


def tuning_jobs():
    return [("single", os.path.join(ROOT, "runs", "fused_tuning", f"lam{lam}", f"fused_p{TUNE_P}_s{s}"),
             ["--net", "1tls_3x3", "--obs-loop", "--reward", "fused", "--fused-lambda", str(lam),
              "--p", str(TUNE_P), "--seed", str(s)], TUNE_P, "validate") for lam in LAMBDAS for s in range(3)]


def choose_lambda():
    if os.path.exists(CHOSEN):
        return json.load(open(CHOSEN))["lambda"]
    rows = []
    for _, out, args, _, _ in tuning_jobs():
        f = os.path.join(out, f"val_stratified_p{TUNE_P}_delay.csv")
        if os.path.exists(f):
            rows.append({"lambda": float(args[args.index("--fused-lambda") + 1]), "seed": out[-1],
                         "val_delay": pd.read_csv(f).delay_all.mean()})
    d = pd.DataFrame(rows)
    per = d.groupby("lambda").val_delay.agg(["mean", "std", "size"]).reset_index()
    best = per.sort_values(["mean", "lambda"]).iloc[0]
    json.dump({"lambda": float(best["lambda"]), "validation": per.to_dict("records"), "rule":
               "lowest mean validation delay over 3 seeds, synthetic p=0.2, 10 episodes"}, open(CHOSEN, "w"), indent=2)
    print(f"chosen lambda = {best['lambda']} | " + "; ".join(f"{r['lambda']}: {r['mean']:.1f}" for _, r in per.iterrows()),
          flush=True)
    return float(best["lambda"])


def main_jobs(lam):
    js = []
    for net in ("cologne1", "ingolstadt1"):
        js += [("single", os.path.join(ROOT, "runs", "fused", net, f"fused_p{p}_s{s}"),
                ["--net", net, "--obs-loop", "--reward", "fused", "--fused-lambda", str(lam), "--p", str(p), "--seed", str(s)],
                p, "test") for p in (0.1, 0.2, 0.4) for s in range(5)]
    js += [("multi", os.path.join(ROOT, "runs", "multi", "cologne3", "isac_auto_loopstate", f"fused_p{p}_s{s}"),
            ["--net", "cologne3", "--obs-loop", "--reward", "fused", "--fused-lambda", str(lam), "--p", str(p), "--seed", str(s)],
            p, "test") for p in (0.1, 0.2) for s in range(3)]
    js += [("single", os.path.join(ROOT, "runs", "fused", "synthetic", f"fused_p{p}_s{s}"),
            ["--net", "1tls_3x3", "--obs-loop", "--reward", "fused", "--fused-lambda", str(lam), "--p", str(p), "--seed", str(s)],
            p, "test") for p in (0.1, 0.2, 0.4) for s in range(5)]
    return js


def heldout_jobs(lam):
    base = os.path.join(ROOT, "runs", "multi", "ingolstadt7")
    js = [("base", os.path.join(base, "baselines"), [b], 0.1, "test")
          for b in ("fixed_time", "max_pressure_cv", "max_pressure_loop")]
    for algo, reward, tag in (("sac", "fused", "isac_auto_loopstate"), ("dqn", "pressure", "idqn_loopstate"),
                              ("sac", "stratified", "isac_auto_loopstate")):
        js += [("multi7", os.path.join(base, tag, f"{reward}_p0.1_s{s}"),
                ["--net", "ingolstadt7", "--algo", algo, "--obs-loop", "--reward", reward, "--fused-lambda", str(lam),
                 "--p", "0.1", "--seed", str(s)], 0.1, "test") for s in range(3)]
    return js


def _call(cmd, log, device):
    with open(log, "a") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True, env=_env(device))


def run(job, device):
    kind, out, args, p, mode = job
    os.makedirs(out, exist_ok=True)
    BUDGET.take(RAM_GB[kind])
    try:
        log = os.path.join(out, "stdout.log")
        if kind == "base":
            if not os.path.exists(os.path.join(out, f"eval_{args[0]}_p{p}.csv")):
                _call([PY, os.path.join(ROOT, "scripts", "evaluate_multi.py"), "--baseline", args[0], "--net", "ingolstadt7",
                       "--p", str(p), "--out", out], log, "cpu")
            return f"{args[0]} ingolstadt7 p={p}"
        multi = kind.startswith("multi")
        train, ev = ("train_multi.py", "evaluate_multi.py") if multi else ("train.py", "evaluate.py")
        if not os.path.exists(os.path.join(out, "DONE")):
            _call([PY, os.path.join(ROOT, "scripts", train), *args, "--out", out, "--device", device], log, device)
        if mode == "validate":
            _call([PY, os.path.join(ROOT, "scripts", "validate.py"), "--run", out, "--episodes", "10", "--delay"], log, "cpu")
        elif not os.path.exists(os.path.join(out, f"eval_agent_p{p}.csv")):
            _call([PY, os.path.join(ROOT, "scripts", ev), "--run", out, "--p", str(p)], log, "cpu")
        return out
    finally:
        BUDGET.give(RAM_GB[kind])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu-workers", type=int, default=3)
    ap.add_argument("--cpu-workers", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    t = tuning_jobs()
    print(f"stage 5: {len(t)} tuning runs, then {len(main_jobs(1.0))} main and {len(heldout_jobs(1.0))} held-out jobs",
          flush=True)
    if args.dry_run:
        return
    run_pool(t, run, args.gpu_workers, args.cpu_workers, "T")
    lam = choose_lambda()
    print("TUNING DONE", flush=True)
    m, h = main_jobs(lam), heldout_jobs(lam)
    # real networks first; the held-out runs share the pool so the big runs start early
    order = m[:36] + h + m[36:]
    run_pool(order, run, args.gpu_workers, args.cpu_workers, "M+H")
    print("stage 5 done", flush=True)


if __name__ == "__main__":
    main()
