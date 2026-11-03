"""Stage 2: ablations and baselines after the main grid (resumable; finished outputs are skipped).

Stage 1, training:
  dqn         Double DQN, rewards privileged/naive/ipw/stratified, p 0.1/0.2/0.4, 3 seeds  (Proposition 4)
  loopstate   SAC + loop-count state channel, rewards privileged/stratified, p 0.1/0.2/0.4, 5 seeds
  uneven      SAC trained with an uneven CV share per approach, rewards privileged/naive/ipw/stratified,
              p 0.2, 3 seeds  (assumption A1)
  fu2026      base paper's QM-ATSC (adapted), p 0.1/0.2/0.4/1.0, 5 seeds
Stage 2, evaluation only:
  cross-p     every main-grid SAC model at the other penetration rates
  demand      every SAC/DQN model, baseline and fu2026 table on fixed and peak demand
  uneven      every main-grid SAC model with an uneven CV share

    python experiments/exp2_ablations.py --gpu-workers 3 --cpu-workers 8
"""

from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
import queue
import subprocess
import sys
import threading

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
P_LEVELS = (0.1, 0.2, 0.4)
UNEVEN = "0.5,1.5,0.75,1.25"
BASELINES = ("fixed_time", "max_pressure_cv", "max_pressure_loop")


def _env(device):
    env = dict(os.environ)
    if device == "cpu":
        env.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", CUDA_VISIBLE_DEVICES="")
    return env


def train_jobs():
    jobs = []
    for r, p, s in itertools.product(("privileged", "naive", "ipw", "stratified"), P_LEVELS, range(3)):
        jobs.append(("rl", os.path.join(ROOT, "runs", "dqn", f"{r}_p{p}_s{s}"),
                     ["--algo", "dqn", "--reward", r, "--p", str(p), "--seed", str(s)], p))
    for r, p, s in itertools.product(("privileged", "stratified"), P_LEVELS, range(5)):
        jobs.append(("rl", os.path.join(ROOT, "runs", "sac_auto_loopstate", f"{r}_p{p}_s{s}"),
                     ["--obs-loop", "--reward", r, "--p", str(p), "--seed", str(s)], p))
    for r, s in itertools.product(("privileged", "naive", "ipw", "stratified"), range(3)):
        jobs.append(("rl_uneven", os.path.join(ROOT, "runs", "sac_auto_uneven", f"{r}_p0.2_s{s}"),
                     ["--p-mult", UNEVEN, "--reward", r, "--p", "0.2", "--seed", str(s)], 0.2))
    for p, s in itertools.product(P_LEVELS + (1.0,), range(5)):
        jobs.append(("fu", os.path.join(ROOT, "runs", "fu2026", f"p{p}_s{s}"), ["--p", str(p), "--seed", str(s)], p))
    return jobs


def run_train(job, device):
    kind, out, args, p = job
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "stdout.log"), "a") as f:
        if kind == "fu":
            if not os.path.exists(os.path.join(out, "DONE")):
                subprocess.run([PY, os.path.join(ROOT, "scripts", "train_fu2026.py"), *args, "--out", out],
                               stdout=f, stderr=subprocess.STDOUT, check=True, env=_env("cpu"))
            res = os.path.join(out, f"eval_fu2026_p{p}.csv")
            if not os.path.exists(res):
                subprocess.run([PY, os.path.join(ROOT, "scripts", "evaluate.py"), "--baseline", "fu2026",
                                "--qtable", os.path.join(out, "qtable.npy"), "--p", str(p), "--out", out],
                               stdout=f, stderr=subprocess.STDOUT, check=True, env=_env("cpu"))
            return out
        if not os.path.exists(os.path.join(out, "DONE")):
            subprocess.run([PY, os.path.join(ROOT, "scripts", "train.py"), *args, "--out", out, "--device", device],
                           stdout=f, stderr=subprocess.STDOUT, check=True, env=_env(device))
        extra = ["--p-mult", UNEVEN] if kind == "rl_uneven" else []
        res = os.path.join(out, f"eval_agent_p{p}" + ("_uneven" if extra else "") + ".csv")
        if not os.path.exists(res):
            subprocess.run([PY, os.path.join(ROOT, "scripts", "evaluate.py"), "--run", out, "--p", str(p), *extra],
                           stdout=f, stderr=subprocess.STDOUT, check=True, env=_env("cpu"))
    return out


def eval_jobs():
    """(output file, evaluate.py arguments)."""
    jobs = []
    main_runs = sorted(d for d in glob.glob(os.path.join(ROOT, "runs", "sac_auto", "*")) if os.path.exists(os.path.join(d, "DONE")))
    all_rl = main_runs + sorted(d for k in ("sac_alpha0.2", "dqn", "sac_auto_loopstate")
                                for d in glob.glob(os.path.join(ROOT, "runs", k, "*")) if os.path.exists(os.path.join(d, "DONE")))
    for d in main_runs:
        p = json.load(open(os.path.join(d, "config.json")))["p"]
        for q in P_LEVELS + (1.0,):
            if q != p:
                jobs.append((os.path.join(d, f"eval_agent_p{q}.csv"), ["--run", d, "--p", str(q)]))
        if p < 1.0:
            jobs.append((os.path.join(d, f"eval_agent_p{p}_uneven.csv"), ["--run", d, "--p", str(p), "--p-mult", UNEVEN]))
    for d in all_rl:
        p = json.load(open(os.path.join(d, "config.json")))["p"]
        for dem in ("fixed", "peak"):
            jobs.append((os.path.join(d, f"eval_agent_p{p}_demand_{dem}.csv"), ["--run", d, "--p", str(p), "--demand", dem]))
    bdir = os.path.join(ROOT, "runs", "baselines")
    for b, p, dem in itertools.product(BASELINES, P_LEVELS + (1.0,), ("fixed", "peak")):
        jobs.append((os.path.join(bdir, f"eval_{b}_p{p}_demand_{dem}.csv"),
                     ["--baseline", b, "--p", str(p), "--demand", dem, "--out", bdir]))
    for b, p in itertools.product(BASELINES, P_LEVELS):
        jobs.append((os.path.join(bdir, f"eval_{b}_p{p}_uneven.csv"),
                     ["--baseline", b, "--p", str(p), "--p-mult", UNEVEN, "--out", bdir]))
    for d in sorted(glob.glob(os.path.join(ROOT, "runs", "fu2026", "*"))):
        if not os.path.exists(os.path.join(d, "DONE")):
            continue
        p = json.load(open(os.path.join(d, "config.json")))["p"]
        q = os.path.join(d, "qtable.npy")
        for dem in ("fixed", "peak"):
            jobs.append((os.path.join(d, f"eval_fu2026_p{p}_demand_{dem}.csv"),
                         ["--baseline", "fu2026", "--qtable", q, "--p", str(p), "--demand", dem, "--out", d]))
        if p < 1.0:
            jobs.append((os.path.join(d, f"eval_fu2026_p{p}_uneven.csv"),
                         ["--baseline", "fu2026", "--qtable", q, "--p", str(p), "--p-mult", UNEVEN, "--out", d]))
    return [j for j in jobs if not os.path.exists(j[0])]


def run_eval(job, device):
    out, args = job
    subprocess.run([PY, os.path.join(ROOT, "scripts", "evaluate.py"), *args], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, env=_env("cpu"))
    return out


def run_pool(jobs, fn, gpu_workers, cpu_workers, label):
    q = queue.Queue()
    for j in jobs:
        q.put(j)
    lock, done = threading.Lock(), [0]

    def worker(device):
        while True:
            try:
                j = q.get_nowait()
            except queue.Empty:
                return
            try:
                msg = f"done {fn(j, device)} [{device}]"
            except subprocess.CalledProcessError as e:
                msg = f"FAILED {e}"
            with lock:
                done[0] += 1
                print(f"[{label} {done[0]}/{len(jobs)}] {msg}", flush=True)

    ts = [threading.Thread(target=worker, args=("cuda",)) for _ in range(gpu_workers)]
    ts += [threading.Thread(target=worker, args=("cpu",)) for _ in range(cpu_workers)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu-workers", type=int, default=3)
    ap.add_argument("--cpu-workers", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    tj = train_jobs()
    print(f"stage 1: {len(tj)} training jobs", flush=True)
    if args.dry_run:
        ej = eval_jobs()
        print(f"stage 2 (now): {len(ej)} evaluation jobs")
        return
    run_pool(tj, run_train, args.gpu_workers, args.cpu_workers, "train")
    ej = eval_jobs()
    print(f"stage 2: {len(ej)} evaluation jobs", flush=True)
    # evaluations are CPU-only and light: use more workers
    run_pool(ej, run_eval, 0, args.gpu_workers + args.cpu_workers + 3, "eval")
    print("stage 2 done", flush=True)


if __name__ == "__main__":
    main()
