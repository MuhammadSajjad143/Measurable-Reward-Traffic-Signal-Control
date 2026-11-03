"""Stage 4: multi-intersection network and pressure-reward (PressLight-style) baselines.

All runs use the final model's state (CV DTSE + loop-count channel). Resumable;
finished outputs are skipped. Jobs run in priority order (the most important
comparisons first), under a RAM budget (a multi-intersection run holds three
replay buffers).

  M. RESCO cologne3 (3 signals, real 07:00-08:00 demand), one agent per signal
       baselines:   fixed-time, max-pressure (CV), max-pressure (loops) x p 0.1/0.2/1.0
       final model: independent SAC x 5 rewards x p 0.1 x 3 seeds, then p 0.2
       PressLight-style: independent DQN + pressure reward x p 0.1/0.2 x 3 seeds
       pressure ablation: independent SAC + pressure reward x p 0.1/0.2 x 3 seeds
  P. Single intersections (synthetic, cologne1, ingolstadt1)
       PressLight-style: DQN + pressure reward x p 0.1/0.2/0.4 x 3 seeds
       pressure ablation: SAC + pressure reward x p 0.1/0.2/0.4 x 3 seeds

    python experiments/exp4_multisignal_pressure.py --gpu-workers 3 --cpu-workers 8
"""

from __future__ import annotations

import argparse
import os
import subprocess
import threading

from exp2_ablations import PY, ROOT, _env, run_pool

REWARDS = ("stratified", "naive", "privileged", "ipw", "loop_count")
BASELINES = ("fixed_time", "max_pressure_cv", "max_pressure_loop")
SINGLE_NETS = ("cologne1", "ingolstadt1", "1tls_3x3")
RAM_GB = {"mbase": 1.0, "multi": 4.0, "single": 3.0}
BUDGET_GB = 26.0


class Budget:
    def __init__(self, total):
        self.free, self.cv = total, threading.Condition()

    def take(self, gb):
        with self.cv:
            self.cv.wait_for(lambda: self.free >= gb)
            self.free -= gb

    def give(self, gb):
        with self.cv:
            self.free += gb
            self.cv.notify_all()


BUDGET = Budget(BUDGET_GB)


def jobs():
    m = os.path.join(ROOT, "runs", "multi", "cologne3")
    js = [("mbase", os.path.join(m, "baselines"), [b], p) for b in BASELINES for p in (0.1, 0.2, 1.0)]

    def multi(algo, r, p, s):
        tag = "idqn_loopstate" if algo == "dqn" else "isac_auto_loopstate"
        return ("multi", os.path.join(m, tag, f"{r}_p{p}_s{s}"),
                ["--algo", algo, "--obs-loop", "--reward", r, "--p", str(p), "--seed", str(s)], p)

    def single(algo, net, p, s):
        tag = ("dqn" if algo == "dqn" else "sac_auto") + "_loopstate"
        base = os.path.join(ROOT, "runs", "pressure", net, tag) if net != "1tls_3x3" else \
            os.path.join(ROOT, "runs", "pressure", "synthetic", tag)
        return ("single", os.path.join(base, f"pressure_p{p}_s{s}"),
                ["--algo", algo, "--net", net, "--obs-loop", "--reward", "pressure", "--p", str(p), "--seed", str(s)], p)

    js += [multi("sac", r, 0.1, s) for s in range(3) for r in REWARDS]
    js += [single("dqn", n, 0.1, s) for n in SINGLE_NETS[:2] for s in range(3)]
    js += [multi(a, "pressure", 0.1, s) for a in ("dqn", "sac") for s in range(3)]
    js += [single(a, n, p, s) for a in ("dqn", "sac") for n in SINGLE_NETS[:2] for p in (0.1, 0.2, 0.4)
           for s in range(3) if not (a == "dqn" and p == 0.1)]
    js += [multi("sac", r, 0.2, s) for s in range(3) for r in REWARDS]
    js += [multi(a, "pressure", 0.2, s) for a in ("dqn", "sac") for s in range(3)]
    js += [single(a, "1tls_3x3", p, s) for a in ("dqn", "sac") for p in (0.1, 0.2, 0.4) for s in range(3)]
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
        if kind == "mbase":
            if not os.path.exists(os.path.join(out, f"eval_{args[0]}_p{p}.csv")):
                _call([PY, os.path.join(ROOT, "scripts", "evaluate_multi.py"), "--baseline", args[0], "--net", "cologne3",
                       "--p", str(p), "--out", out], log, "cpu")
            return f"{args[0]} cologne3 p={p}"
        train, ev = ("train_multi.py", "evaluate_multi.py") if kind == "multi" else ("train.py", "evaluate.py")
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
    print(f"stage 4: {len(js)} jobs ({sum(j[0] == 'multi' for j in js)} multi-intersection runs, "
          f"{sum(j[0] == 'single' for j in js)} single-intersection runs)", flush=True)
    if args.dry_run:
        return
    run_pool(js, run, args.gpu_workers, args.cpu_workers, "X3")
    print("stage 4 done", flush=True)


if __name__ == "__main__":
    main()
