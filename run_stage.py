"""Runs every run of one stage for one algorithm, several at a time, without running out of memory.

    python run_stage.py --algo TD3 --stage selection      # 3 Part C candidates x seeds 21-23 = 9 runs
    python select_config.py --algo TD3                    # picks the configuration (writes selected_TD3.json)
    python run_stage.py --algo TD3 --stage final          # the selected configuration x seeds 31-35 = 5 runs
    python run_stage.py --algo PPO --stage selection --normalize   # supplementary PPO comparison (results_ppo_norm/)

Runs that already have a summary.json are skipped, so the command can simply be started again after an interruption
(an unfinished run starts again from the beginning). A run starts only while fewer than --max-parallel are running and
at least --min-free-gb of memory is available (default: 4 GB, or a quarter of the memory on smaller machines; the
first run needs only --stop-gb + 1 GB);
below --stop-gb the newest run is paused until --stop-gb + 1 GB is available again.
Logs: results/<stage>/logs/.
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

parser = argparse.ArgumentParser()
parser.add_argument("--algo", choices=["PPO", "SAC", "TD3"], required=True)
parser.add_argument("--stage", choices=list(STAGES), required=True)
parser.add_argument("--max-parallel", type=int, default=max(1, min(6, (os.cpu_count() or 2) - 1)))
TOTAL_GB = psutil.virtual_memory().total / 2**30                    # one run needs about 0.2-0.8 GB
parser.add_argument("--min-free-gb", type=float, default=min(4.0, 0.25 * TOTAL_GB))
parser.add_argument("--stop-gb", type=float, default=min(2.0, 0.125 * TOTAL_GB))
parser.add_argument("--gap", type=float, default=60.0, help="seconds between two starts (5 with --smoke)")
parser.add_argument("--smoke", action="store_true", help="3,000-step code check of the whole pipeline")
parser.add_argument("--normalize", action="store_true", help="supplementary comparison: PPO with VecNormalize")
parser.add_argument("--equal-time", action="store_true", help="supplementary comparison: SAC's training time (final stage)")
args = parser.parse_args()
if args.smoke and "--gap" not in sys.argv:
    args.gap = 5.0
if args.normalize and args.algo != "PPO":
    parser.error("--normalize is the supplementary PPO comparison")
ROOT = results_root(args.smoke, args.normalize, args.equal_time)

if args.stage == "selection":
    configs = [c for c in CONFIGS if c.startswith(args.algo)]
else:
    chosen = ROOT / "selection" / f"selected_{args.algo}.json"
    if not chosen.exists():
        sys.exit(f"{chosen} not found: run the selection stage and select_config.py --algo {args.algo} first")
    configs = [json.load(open(chosen))["selected"]]
jobs = [(c, s) for c in configs for s in STAGES[args.stage]["seeds"]
        if not (run_dir(c, s, args.stage, args.smoke, args.normalize, args.equal_time) / "summary.json").exists()]
logs = ROOT / args.stage / "logs"
logs.mkdir(parents=True, exist_ok=True)
free = lambda: psutil.virtual_memory().available / 2**30
log = lambda msg: print(time.strftime("%H:%M:%S"), msg, flush=True)
log(f"{args.algo} {args.stage}: {len(jobs)} runs to do ({', '.join(f'{c} seed {s}' for c, s in jobs) or 'none'})")
running, paused, last_start, t0, last_wait = [], [], 0.0, time.time(), 0.0

while jobs or running:
    for proc, name in list(running):
        if proc.poll() is not None:
            running.remove((proc, name))
            log(f"done: {name} (exit {proc.returncode}{', see log' if proc.returncode else ''}) | free {free():.1f} GB")
    if free() < args.stop_gb and running:                              # memory emergency: pause the newest run
        proc, name = running[-1]
        if proc.pid not in paused:
            psutil.Process(proc.pid).suspend()                      # works on macOS, Linux and Windows
            paused.append(proc.pid)
            log(f"paused {name}: free {free():.1f} GB")
    elif paused and free() > args.stop_gb + 1.0:                      # a paused run keeps its memory, so resuming needs little
        pid = paused.pop()
        psutil.Process(pid).resume()
        log(f"resumed pid {pid}: free {free():.1f} GB")
    # one run needs well under 1 GB, so the first run may start below --min-free-gb (but 1 GB above --stop-gb)
    room = free() >= args.min_free_gb or (not running and free() >= args.stop_gb + 1.0)
    if jobs and not paused and len(running) < args.max_parallel and room and time.time() - last_start >= args.gap:
        config, seed = jobs.pop(0)
        name = f"{config}_seed{seed}"
        cmd = [sys.executable, str(HERE / "train_eval.py"), "--config", config, "--seed", str(seed), "--stage", args.stage]
        out = open(logs / f"{name}.log", "w")
        flags = (["--smoke"] if args.smoke else []) + (["--normalize"] if args.normalize else []) + (["--equal-time"] if args.equal_time else [])
        proc = subprocess.Popen(cmd + flags, stdout=out, stderr=subprocess.STDOUT, cwd=HERE)
        running.append((proc, name))
        last_start = time.time()
        log(f"started: {name} | running {len(running)} | free {free():.1f} GB")
    elif jobs and not running and time.time() - last_wait >= 300:   # nothing can start: say why instead of waiting silently
        last_wait = time.time()
        log(f"waiting for memory: free {free():.1f} GB, the first run starts at {args.stop_gb + 1.0:.1f} GB (close other apps)")
    time.sleep(5)
log(f"finished in {(time.time() - t0) / 60:.0f} min")
