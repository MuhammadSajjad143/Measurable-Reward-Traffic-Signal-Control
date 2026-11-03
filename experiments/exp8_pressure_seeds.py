"""Stage 8: five seeds for the pressure-reward learners on the single intersections.

The PressLight-style DQN and SAC with the pressure reward had seeds 0-2 (stage 4); this adds
seeds 3 and 4 with the same settings, so every single-intersection learner has five seeds.
Each new run gets the same evaluations its seeds 0-2 had:
  at its training p (0.1 / 0.2 / 0.4), and at p = 0.1
  DQN: loop noise 5/10/20 %, one failed loop, and (real networks) demand x1.3 / x1.6
  SAC (real networks): demand x1.3 / x1.6

    python experiments/exp8_pressure_seeds.py --gpu-workers 3 --cpu-workers 8
"""

from __future__ import annotations

import argparse
import os

from exp2_ablations import ROOT, run_pool
from exp7_final_model_tests import NETS, run

SEEDS = (3, 4)


def pressure_dir(algo, net, p, s):
    tag = ("dqn" if algo == "dqn" else "sac_auto") + "_loopstate"
    return os.path.join(ROOT, "runs", "pressure", "synthetic" if net == "1tls_3x3" else net, tag, f"pressure_p{p}_s{s}")


def train_jobs():
    return [("single", pressure_dir(algo, net, p, s),
             ["--algo", algo, "--net", net, "--obs-loop", "--reward", "pressure", "--p", str(p), "--seed", str(s)])
            for p in (0.1, 0.2, 0.4) for net in NETS for algo in ("dqn", "sac") for s in SEEDS]


def eval_jobs():
    js = []
    for net in NETS:
        real = net != "1tls_3x3"
        for s in SEEDS:
            d = pressure_dir("dqn", net, 0.1, s)
            for cond in (["--loop-noise", "0.05"], ["--loop-noise", "0.1"], ["--loop-noise", "0.2"], ["--loop-fail"]):
                js.append(("eval", ["--run", d, "--p", "0.1", *cond]))
            if real:
                for algo in ("dqn", "sac"):
                    for sc in ("1.3", "1.6"):
                        js.append(("eval", ["--run", pressure_dir(algo, net, 0.1, s), "--p", "0.1", "--demand-scale", sc]))
    return js


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu-workers", type=int, default=3)
    ap.add_argument("--cpu-workers", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    t, e = train_jobs(), eval_jobs()
    print(f"stage 8: {len(t)} training runs, then {len(e)} evaluations", flush=True)
    if args.dry_run:
        return
    run_pool(t, run, args.gpu_workers, args.cpu_workers, "X7-train")
    print("TRAINING DONE", flush=True)
    run_pool(e, run, 0, args.gpu_workers + args.cpu_workers, "X7-eval")
    print("stage 8 done", flush=True)


if __name__ == "__main__":
    main()
