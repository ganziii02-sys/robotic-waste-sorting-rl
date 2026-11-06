"""Final results of every algorithm found in results/final/ (copy teammates' run folders there to compare).

    python report_results.py

Prints, per algorithm, the mean and standard deviation over the final seeds of each test, and saves
results/final/final_results.csv. Uses summary_reevaluated.json when a run was evaluated again on one machine.
--normalize reports the supplementary PPO comparison (results_ppo_norm/).
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from train_eval import PROTOCOL, results_root  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--smoke", action="store_true")
parser.add_argument("--normalize", action="store_true")
parser.add_argument("--equal-time", action="store_true")
args = parser.parse_args()
pd.set_option("display.width", 220)

rows = []
ROOT = results_root(args.smoke, args.normalize, args.equal_time)
for folder in sorted((ROOT / "final").glob("*_seed*")):
    files = [folder / "summary_reevaluated.json", folder / "summary.json"]
    path = next((f for f in files if f.exists()), None)
    if path is None:
        continue
    s = json.load(open(path))
    if s["protocol"] != PROTOCOL:
        print("skipped (other protocol):", folder.name)
        continue
    row = {"algorithm": s["algorithm"] + (" + norm" if s.get("observation and reward normalisation") else "")
                        + (f" {s['steps'] / 1e6:.2f}M steps" if s.get("equal-time budget") or s.get("equal-time budget (4M steps)") else ""),
           "config": s["config"], "seed": s["seed"], "evaluated": path.stem,
           "nominal reached %": s["test: nominal"]["reached %"], "nominal lifted %": s["test: nominal"]["lifted %"]}
    for key, value in s.items():
        if key.startswith("test: "):
            row[key[6:] + " success %"] = value["success %"]
    for key in ("tray clearing (20 trays, nominal)", "dense tray clearing (20 trays, nominal)"):
        label = "dense tray" if key.startswith("dense") else "tray"
        for n, t in s[key].items():
            row[f"{label} {n}: per-item %"] = t["per-item success %"]
            row[f"{label} {n}: whole %"] = t["whole-tray success %"]
    ok = s["test: nominal"]["movement (successful episodes)"]
    row["steps per success"] = ok["steps"] if ok else float("nan")
    rows.append(row)
if not rows:
    sys.exit("no final runs found in " + str(ROOT / "final"))

df = pd.DataFrame(rows)
df.to_csv(ROOT / "final" / "final_results.csv", index=False)
print(df[["algorithm", "config", "seed", "evaluated", "nominal success %"]].to_string(index=False))
metrics = [c for c in df.columns if c not in ("algorithm", "config", "seed", "evaluated")]
table = df.groupby("algorithm")[metrics].agg(["mean", "std"])
print("\nmean (sd) over final seeds:")
for algo, r in table.iterrows():
    n = int((df.algorithm == algo).sum())
    print(f"\n{algo} ({df[df.algorithm == algo].config.iloc[0]}, {n} seeds)")
    for m in metrics:
        print(f"  {m:<40} {r[(m, 'mean')]:6.1f} ({r[(m, 'std')]:.1f})")
