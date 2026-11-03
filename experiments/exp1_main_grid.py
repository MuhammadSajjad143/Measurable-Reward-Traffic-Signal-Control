"""Stage 1: the main experiment grid, run in parallel (train, then evaluate, per run).

Each job is skipped if its outputs already exist, so the grid can be resumed.

    python experiments/exp1_main_grid.py --grid main --workers 12 --dry-run
    python experiments/exp1_main_grid.py --grid main --workers 12
"""

from __future__ import annotations

import argparse
import itertools
import os
import subprocess
import sys
import queue
import threading

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable

P_LEVELS = (0.1, 0.2, 0.4)
CV_REWARDS = ("privileged", "naive", "ipw", "stratified", "loop_count")
BASELINES = ("fixed_time", "max_pressure_cv", "max_pressure_loop")


def grid(name):
    """List of (alpha, reward, p, seed)."""
    jobs = []
    if name in ("main", "all"):
        # every reward at every partial-detection level, automatic temperature
        jobs += [("auto", r, p, s) for r, p, s in itertools.product(CV_REWARDS, P_LEVELS, range(5))]
        # full-detection reference (all CV estimators equal the privileged one at p = 1)
        jobs += [("auto", r, 1.0, s) for r, s in itertools.product(("privileged", "loop_count"), range(5))]
    if name in ("temperature", "all"):
        # fixed temperature, where the reward scale should act like alpha / p
        jobs += [("0.2", r, p, s) for r, p, s in itertools.product(("privileged", "naive", "ipw"), P_LEVELS, range(3))]
    if name == "pilot":
        jobs += [("auto", r, 0.2, 0) for r in ("privileged", "naive", "stratified")]
    return jobs


def run_dir(alpha, reward, p, seed):
    tag = "auto" if alpha == "auto" else f"alpha{alpha}"
    return os.path.join(ROOT, "runs", f"sac_{tag}", f"{reward}_p{p}_s{seed}")


def job(alpha, reward, p, seed, steps, device="cuda"):
    out = run_dir(alpha, reward, p, seed)
    log = os.path.join(out, "stdout.log")
    os.makedirs(out, exist_ok=True)
    with open(log, "a") as f:
        if not os.path.exists(os.path.join(out, "DONE")):
            subprocess.run([PY, os.path.join(ROOT, "scripts", "train.py"), "--reward", reward, "--p", str(p),
                            "--seed", str(seed), "--alpha", alpha, "--steps", str(steps), "--out", out,
                            "--device", device],
                           stdout=f, stderr=subprocess.STDOUT, check=True, env=_env(device))
        if not os.path.exists(os.path.join(out, f"eval_agent_p{p}.csv")):
            subprocess.run([PY, os.path.join(ROOT, "scripts", "evaluate.py"), "--run", out, "--p", str(p)],
                           stdout=f, stderr=subprocess.STDOUT, check=True)
    return out


def _env(device):
    env = dict(os.environ)
    if device == "cpu":  # 2 math threads per CPU run; SUMO uses one more core
        env.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", CUDA_VISIBLE_DEVICES="")
    return env


def baseline_job(name, p):
    out = os.path.join(ROOT, "runs", "baselines")
    if os.path.exists(os.path.join(out, f"eval_{name}_p{p}.csv")):
        return out
    subprocess.run([PY, os.path.join(ROOT, "scripts", "evaluate.py"), "--baseline", name, "--p", str(p),
                    "--out", out], check=True, stdout=subprocess.DEVNULL)
    return f"{name} p={p}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="main", choices=["main", "temperature", "all", "pilot"])
    ap.add_argument("--workers", type=int, default=None, help="(old) same as --gpu-workers")
    ap.add_argument("--gpu-workers", type=int, default=3)
    ap.add_argument("--cpu-workers", type=int, default=8)
    ap.add_argument("--steps", type=int, default=180_000)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.workers is not None:
        args.gpu_workers, args.cpu_workers = args.workers, 0

    jobs = grid(args.grid)
    print(f"{len(jobs)} training runs, {args.gpu_workers} GPU + {args.cpu_workers} CPU workers", flush=True)
    if args.dry_run:
        for j in jobs:
            print(j, run_dir(*j))
        return

    for b in BASELINES:
        for p in P_LEVELS + (1.0,):
            baseline_job(b, p)

    # one shared queue; every worker thread owns a device
    q = queue.Queue()
    for j in jobs:
        q.put(j)
    lock, counter = threading.Lock(), [0]

    def worker(device):
        while True:
            try:
                j = q.get_nowait()
            except queue.Empty:
                return
            try:
                msg = f"done {job(*j, args.steps, device)} [{device}]"
            except subprocess.CalledProcessError as e:
                msg = f"FAILED {e}"
            with lock:
                counter[0] += 1
                print(f"[{counter[0]}/{len(jobs)}] {msg}", flush=True)

    threads = [threading.Thread(target=worker, args=("cuda",)) for _ in range(args.gpu_workers)]
    threads += [threading.Thread(target=worker, args=("cpu",)) for _ in range(args.cpu_workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


if __name__ == "__main__":
    main()
