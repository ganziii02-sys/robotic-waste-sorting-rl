"""Selection stage: picks one configuration per algorithm with the rule fixed in Part C.

    python select_config.py --algo TD3

Ranks the three Part C candidates of the algorithm by their mean over the selection-stage training seeds of the selection episodes
(seeds 1000-1049, never used as test episodes): highest success rate, then highest lift rate, then highest mean episode
reward. Only complete runs made with this package's protocol count. Writes results/selection/selected_<ALGO>.json
(--normalize: the supplementary PPO comparison in results_ppo_norm/).
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from train_eval import CONFIGS, PROTOCOL, STAGES, STEPS, results_root, run_dir  # noqa: E402

KEY = "selection: nominal (seeds 1000-1049)"
parser = argparse.ArgumentParser()
parser.add_argument("--algo", choices=["PPO", "SAC", "TD3"], required=True)
parser.add_argument("--smoke", action="store_true")
parser.add_argument("--normalize", action="store_true")
args = parser.parse_args()
pd.set_option("display.width", 200)

rows, problems = [], []
candidates = [c for c in CONFIGS if c.startswith(args.algo)]
for config in candidates:
    for seed in STAGES["selection"]["seeds"]:
        path = run_dir(config, seed, "selection", args.smoke, args.normalize) / "summary.json"
        if not path.exists():
            problems.append(f"missing: {config} seed {seed}")
            continue
        s = json.load(open(path))
        if (s["protocol"] != PROTOCOL or s["smoke test (not a result)"] != args.smoke or (not args.smoke and s["steps"] < STEPS)
                or bool(s.get("observation and reward normalisation")) != args.normalize):
            problems.append(f"not a valid run of this protocol: {path}")
            continue
        r = s[KEY]
        rows.append({"config": config, "seed": seed, "success %": r["success %"], "lifted %": r["lifted %"],
                     "mean reward": r["mean reward"]})
if problems:
    sys.exit("selection not possible yet:\n  " + "\n  ".join(problems))

df = pd.DataFrame(rows)
print(df.round(2).to_string(index=False))
mean = df.groupby("config")[["success %", "lifted %", "mean reward"]].mean()
ranked = sorted(candidates, key=lambda c: tuple(mean.loc[c]), reverse=True)
seeds = "-".join(str(x) for x in (STAGES["selection"]["seeds"][0], STAGES["selection"]["seeds"][-1]))
print(f"\nmean over seeds {seeds} (ranked):")
print(mean.loc[ranked].round(2).to_string())
out = results_root(args.smoke, args.normalize) / "selection" / f"selected_{args.algo}.json"
json.dump({"selected": ranked[0], "rule": f"mean over training seeds {seeds} of selection-episode success, then lift, then reward",
           "means": mean.loc[ranked].to_dict(orient="index")}, open(out, "w"), indent=2)
print(f"\nselected {args.algo} configuration: {ranked[0]}  (saved {out.relative_to(HERE)})")
