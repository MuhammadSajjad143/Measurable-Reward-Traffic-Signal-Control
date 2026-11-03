"""Stage 6: same-signal (collapsed-strata) fallback for lanes without a CV, motivated by the fairness
gap on ingolstadt7, and a second held-out network.

Reward fused_cs = fused reward of stage 5 (lambda* from runs/fused_tuning/CHOSEN.json, not
re-tuned) whose stratified cost uses the collapsed-strata fallback: a lane without a CV borrows
the mean CV cost of the lanes with the same signal indication. Nothing else changes.

  H. New held-out network, never used before: RESCO cologne8 (8 signals), p = 0.1
       baselines; SAC fused_cs, SAC fused (old fallback), PressLight-style DQN; 3 seeds each
  D. ingolstadt7, where the gap was diagnosed: SAC fused_cs, 3 seeds (old fused already run)
  S. Safety check, p = 0.1: fused_cs on cologne3 (3 seeds) and on cologne1, ingolstadt1 and
     the synthetic intersection (5 seeds)

    python experiments/exp6_same_signal_fallback.py --gpu-workers 3 --cpu-workers 8
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import threading

from exp2_ablations import PY, ROOT, _env, run_pool

RAM_GB = {"base": 2.0, "single": 3.0, "multi": 4.0, "multi7": 9.0}
MAX_GB, SAFETY_GB = 28.0, 3.0


def _external_gb():
    """RAM used by everything except our training and evaluation processes (e.g. other programs on the machine)."""
    info = dict(line.split(":", 1) for line in open("/proc/meminfo"))
    total = int(info["MemTotal"].split()[0]) / 1e6
    avail = int(info["MemAvailable"].split()[0]) / 1e6
    ours = subprocess.run(["bash", "-c", "ps -eo rss,args | grep -E 'scripts/(tr[a]in|ev[a]luate)' | awk '{s+=$1} END {print s+0}'"],
                          capture_output=True, text=True).stdout.strip()
    return total, total - avail - float(ours or 0) / 1e6


class MemoryAwareBudget:
    """Reserve RAM for a job only if it fits next to whatever else is running on the machine."""

    def __init__(self):
        self.reserved, self.cv = 0.0, threading.Condition()

    def take(self, gb):
        with self.cv:
            while True:
                total, external = _external_gb()
                if self.reserved + gb <= min(MAX_GB, total - external - SAFETY_GB):
                    self.reserved += gb
                    return
                self.cv.wait(30)

    def give(self, gb):
        with self.cv:
            self.reserved -= gb
            self.cv.notify_all()


BUDGET = MemoryAwareBudget()
LAM = json.load(open(os.path.join(ROOT, "runs", "fused_tuning", "CHOSEN.json")))["lambda"]


def jobs():
    js = []
    c8 = os.path.join(ROOT, "runs", "multi", "cologne8")
    js += [("base", os.path.join(c8, "baselines"), ["cologne8", b], 0.1)
           for b in ("fixed_time", "max_pressure_cv", "max_pressure_loop")]
    held = [("sac", "fused_cs", "isac_auto_loopstate"), ("sac", "fused", "isac_auto_loopstate"),
            ("dqn", "pressure", "idqn_loopstate")]
    for s in range(3):
        for algo, reward, tag in held:
            js.append(("multi7", os.path.join(c8, tag, f"{reward}_p0.1_s{s}"),
                       ["--net", "cologne8", "--algo", algo, "--obs-loop", "--reward", reward,
                        "--fused-lambda", str(LAM), "--p", "0.1", "--seed", str(s)], 0.1))
        js.append(("multi7", os.path.join(ROOT, "runs", "multi", "ingolstadt7", "isac_auto_loopstate", f"fused_cs_p0.1_s{s}"),
                   ["--net", "ingolstadt7", "--obs-loop", "--reward", "fused_cs", "--fused-lambda", str(LAM),
                    "--p", "0.1", "--seed", str(s)], 0.1))
    js += [("multi", os.path.join(ROOT, "runs", "multi", "cologne3", "isac_auto_loopstate", f"fused_cs_p0.1_s{s}"),
            ["--net", "cologne3", "--obs-loop", "--reward", "fused_cs", "--fused-lambda", str(LAM),
             "--p", "0.1", "--seed", str(s)], 0.1) for s in range(3)]
    for net in ("cologne1", "ingolstadt1", "1tls_3x3"):
        tag = "synthetic" if net == "1tls_3x3" else net
        js += [("single", os.path.join(ROOT, "runs", "fused_cs", tag, f"fused_cs_p0.1_s{s}"),
                ["--net", net, "--obs-loop", "--reward", "fused_cs", "--fused-lambda", str(LAM),
                 "--p", "0.1", "--seed", str(s)], 0.1) for s in range(5)]
    return js


def _call(cmd, log, device):
    with open(log, "a") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True, env=_env(device))


def run(job, device):
    kind, out, args, p = job
    os.makedirs(out, exist_ok=True)
    BUDGET.take(RAM_GB[kind])
    try:
        log = os.path.join(out, "stdout.log")
        if kind == "base":
            net, name = args
            if not os.path.exists(os.path.join(out, f"eval_{name}_p{p}.csv")):
                _call([PY, os.path.join(ROOT, "scripts", "evaluate_multi.py"), "--baseline", name, "--net", net,
                       "--p", str(p), "--out", out], log, "cpu")
            return f"{name} {net} p={p}"
        multi = kind.startswith("multi")
        train, ev = ("train_multi.py", "evaluate_multi.py") if multi else ("train.py", "evaluate.py")
        if not os.path.exists(os.path.join(out, "DONE")):
            _call([PY, os.path.join(ROOT, "scripts", train), *args, "--out", out, "--device", device], log, device)
        if not os.path.exists(os.path.join(out, f"eval_agent_p{p}.csv")):
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
    js = jobs()
    print(f"stage 6: lambda = {LAM}; {len(js)} jobs "
          f"({sum(j[0] == 'multi7' for j in js)} large multi-signal runs)", flush=True)
    if args.dry_run:
        return
    run_pool(js, run, args.gpu_workers, args.cpu_workers, "X5")
    print("stage 6 done", flush=True)


if __name__ == "__main__":
    main()
