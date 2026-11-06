"""Runs the whole final experiment on one machine: the selection stage of every group, the configuration choice, then
the final runs, several runs at a time, without running out of memory.

    python run_all.py                       # main comparison (PPO, SAC, TD3) + supplementary PPO with VecNormalize
    python run_all.py --smoke               # 3,000-step check of the whole pipeline

Groups: SAC, TD3, PPO (main comparison) and PPO + normalisation (supplementary, results_ppo_norm/). Runs with a
summary.json are skipped, so the command can be started again after an interruption. A run starts only while fewer
than --max-parallel are running and at least --min-free-gb is available (the first run needs --stop-gb + 1 GB). Below
--stop-gb the newest running run is paused, then the next newest, and so on; paused runs resume one at a time, oldest
first, once --stop-gb + 1 GB is available. Log: results/run_all.log (a line per event).
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import psutil

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from train_eval import CONFIGS, STAGES, results_root, run_dir  # noqa: E402

GROUPS = [("SAC", False), ("TD3", False), ("PPO", False), ("PPO", True)]     # slowest first
parser = argparse.ArgumentParser()
parser.add_argument("--max-parallel", type=int, default=max(1, min(6, (os.cpu_count() or 2) - 1)))
TOTAL_GB = psutil.virtual_memory().total / 2**30
parser.add_argument("--min-free-gb", type=float, default=min(4.0, 0.25 * TOTAL_GB))
parser.add_argument("--stop-gb", type=float, default=min(2.0, 0.125 * TOTAL_GB))
parser.add_argument("--gap", type=float, default=60.0, help="seconds between two starts (5 with --smoke)")
parser.add_argument("--smoke", action="store_true")
args = parser.parse_args()
if args.smoke and "--gap" not in sys.argv:
    args.gap = 5.0
python = sys.executable
free = lambda: psutil.virtual_memory().available / 2**30
results_root(args.smoke).mkdir(parents=True, exist_ok=True)
LOG = open(results_root(args.smoke) / "run_all.log", "a")


def log(msg):
    line = f"{time.strftime('%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    LOG.write(line + "\n")
    LOG.flush()


def label(algo, norm):
    return algo + (" + norm" if norm else "")


def jobs_for(stage):
    jobs = []
    for algo, norm in GROUPS:
        if stage == "selection":
            configs = [c for c in CONFIGS if c.startswith(algo)]
        else:
            configs = [json.load(open(results_root(args.smoke, norm) / "selection" / f"selected_{algo}.json"))["selected"]]
        for config in configs:
            for seed in STAGES[stage]["seeds"]:
                if not (run_dir(config, seed, stage, args.smoke, norm) / "summary.json").exists():
                    jobs.append((config, seed, stage, norm))
    return jobs


def run(jobs):
    running, paused, last_start, last_wait = [], [], 0.0, 0.0       # running: (proc, name), oldest first
    while jobs or running:
        for proc, name in list(running):
            if proc.poll() is not None:
                running.remove((proc, name))
                if proc.pid in paused:
                    paused.remove(proc.pid)
                log(f"done: {name} (exit {proc.returncode}{', see its log' if proc.returncode else ''}) | free {free():.1f} GB")
        active = [(p, n) for p, n in running if p.pid not in paused]
        if free() < args.stop_gb and len(active) > 1:                   # pause the newest active run (keep one going)
            proc, name = active[-1]
            psutil.Process(proc.pid).suspend()
            paused.append(proc.pid)
            log(f"paused {name}: free {free():.1f} GB")
        elif paused and free() > args.stop_gb + 1.0:                    # resume the oldest paused run
            pid = min(paused, key=lambda p: [q.pid for q, _ in running].index(p))
            psutil.Process(pid).resume()
            paused.remove(pid)
            log(f"resumed pid {pid}: free {free():.1f} GB")
        room = free() >= args.min_free_gb or (not running and free() >= args.stop_gb + 1.0)
        if jobs and not paused and len(running) < args.max_parallel and room and time.time() - last_start >= args.gap:
            config, seed, stage, norm = jobs.pop(0)
            name = f"{stage}/{config}{'+norm' if norm else ''}_seed{seed}"
            logs = results_root(args.smoke, norm) / stage / "logs"
            logs.mkdir(parents=True, exist_ok=True)
            cmd = [python, str(HERE / "train_eval.py"), "--config", config, "--seed", str(seed), "--stage", stage]
            cmd += (["--smoke"] if args.smoke else []) + (["--normalize"] if norm else [])
            proc = subprocess.Popen(cmd, stdout=open(logs / f"{config}_seed{seed}.log", "w"), stderr=subprocess.STDOUT, cwd=HERE)
            running.append((proc, name))
            last_start = time.time()
            log(f"started: {name} | running {len(running)} | free {free():.1f} GB")
        elif jobs and not running and time.time() - last_wait >= 300:
            last_wait = time.time()
            log(f"waiting for memory: free {free():.1f} GB, the first run starts at {args.stop_gb + 1.0:.1f} GB")
        time.sleep(5)


t0 = time.time()
selection = jobs_for("selection")
log(f"selection stage: {len(selection)} runs to do")
run(selection)
for algo, norm in GROUPS:
    cmd = [python, str(HERE / "select_config.py"), "--algo", algo] + (["--smoke"] if args.smoke else []) + (["--normalize"] if norm else [])
    out = subprocess.run(cmd, capture_output=True, text=True, cwd=HERE)
    chosen = [l for l in out.stdout.splitlines() if l.startswith("selected")]
    log(f"{label(algo, norm)}: {chosen[0] if chosen else 'SELECTION FAILED: ' + (out.stdout + out.stderr)[-500:]}")
    if not chosen:
        sys.exit(1)
final = jobs_for("final")
log(f"final runs: {len(final)} runs to do")
run(final)
log(f"all finished in {(time.time() - t0) / 3600:.1f} h")
