"""One final-stage run: train one Part C candidate configuration with one seed, then run the full evaluation.

    python train_eval.py --config TD3_default_like --seed 21 --stage selection     # selection stage: seeds 21, 22, 23
    python train_eval.py --config TD3_default_like --seed 31 --stage final         # final runs: seeds 31-35
    python train_eval.py --config PPO_small_lr --seed 21 --stage selection --normalize   # supplementary: PPO + VecNormalize
    python train_eval.py --config PPO_small_lr --seed 31 --stage final --equal-time      # supplementary: SAC's training time
    python train_eval.py --config TD3_default_like --seed 21 --stage selection --smoke   # 3,000-step code check only
    python train_eval.py --evaluate-only results/final/TD3_default_like_seed31     # evaluate a saved model again

Fixed for every algorithm (the team's fairness rules): the environment in waste_env.py with all final options,
256x256 hidden layers (PPO: separate actor and critic), 1,000,000 environment steps, one reverse curriculum shared by
all of a run's training environments, no observation or reward normalisation, CPU and one torch thread, the final
model (no checkpoint picking), deterministic actions in evaluation. Only the nine Part C candidates can be run; SAC and
TD3 keep their whole history in the replay buffer (1,000,000 transitions, the library default and the original papers').

Supplementary comparison (--normalize, PPO only): the same procedure with observation and reward normalisation
(VecNormalize), a common PPO implementation practice. Its runs go to results_ppo_norm/ and are reported separately; the
statistics saved during training are used, frozen, in every evaluation.

Evaluation (normal start; test seeds 100-149 in the selection stage, 3000-3049 in the final runs): one item with the
nominal properties, with the training randomisation and with each out-of-range condition; selection episodes (seeds
1000-1049, used only by select_config.py); trays of 3 and 4 items cleared outermost-first, normal and dense (items
0-1 cm apart), 20 trays each; test success of the checkpoints saved every 25k steps; diagnostics.
Results: results/<stage>/<config>_seed<seed>/ (summary.json, test_nominal.csv, model_final.zip, checkpoints/).
"""
import argparse
import json
import os
import platform
import random
import sys
import time
from importlib import metadata
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import waste_env as V  # noqa: E402

PROTOCOL = "3H-final-v2"
STEPS = 1_000_000
# supplementary: SAC's training time for 1,000,000 steps (M5, final runs: SAC/PPO time per step 3.96, SAC/TD3 1.39)
STEPS_EQUAL_TIME = {"PPO": 4_000_000, "TD3": 1_390_000}
STAGES = {"selection": {"seeds": (21, 22, 23), "test_start": 100}, "final": {"seeds": (31, 32, 33, 34, 35), "test_start": 3000}}
SELECTION_SEEDS = range(1000, 1050)
CONFIGS = {   # the Part C candidates (three per algorithm); replay buffer = whole history (library default)
    "PPO_default_like": dict(learning_rate=3e-4, n_steps=512, batch_size=64, n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.0),
    "PPO_small_lr": dict(learning_rate=1e-4, n_steps=512, batch_size=128, n_epochs=10, gamma=0.98, gae_lambda=0.95, clip_range=0.2, ent_coef=0.0),
    "PPO_more_exploration": dict(learning_rate=3e-4, n_steps=256, batch_size=64, n_epochs=10, gamma=0.98, gae_lambda=0.95, clip_range=0.2, ent_coef=0.005),
    "SAC_default_like": dict(learning_rate=3e-4, buffer_size=1_000_000, learning_starts=1_000, batch_size=256, gamma=0.99, tau=0.005, train_freq=1, gradient_steps=1, ent_coef="auto", target_entropy="auto"),
    "SAC_robotics_gamma": dict(learning_rate=3e-4, buffer_size=1_000_000, learning_starts=2_000, batch_size=256, gamma=0.95, tau=0.02, train_freq=1, gradient_steps=1, ent_coef="auto", target_entropy="auto"),
    "SAC_more_exploration": dict(learning_rate=3e-4, buffer_size=1_000_000, learning_starts=2_000, batch_size=256, gamma=0.98, tau=0.01, train_freq=1, gradient_steps=1, ent_coef="auto", target_entropy=-2.0),
    "TD3_default_like": dict(learning_rate=1e-3, buffer_size=1_000_000, learning_starts=1_000, batch_size=256, gamma=0.99, tau=0.005, train_freq=1, gradient_steps=1, noise_sigma=0.10),
    "TD3_robotics_gamma": dict(learning_rate=3e-4, buffer_size=1_000_000, learning_starts=2_000, batch_size=256, gamma=0.95, tau=0.02, train_freq=1, gradient_steps=1, noise_sigma=0.10),
    "TD3_more_exploration": dict(learning_rate=3e-4, buffer_size=1_000_000, learning_starts=2_000, batch_size=256, gamma=0.98, tau=0.01, train_freq=1, gradient_steps=1, noise_sigma=0.20),
}
N_ENVS = {"PPO": 4, "SAC": 1, "TD3": 1}
ENV_FLAGS = dict(grasp_reward=True, regrasp=True, approach_stages=False, relative_obs=True, visual=False)
ALGOS = {"PPO": PPO, "SAC": SAC, "TD3": TD3}


def results_root(smoke, normalize=False, equal_time=False):
    return HERE / (("results_smoke" if smoke else "results") + ("_ppo_norm" if normalize else "") + ("_equal_time" if equal_time else ""))


def run_dir(config, seed, stage, smoke=False, normalize=False, equal_time=False):
    return results_root(smoke, normalize, equal_time) / stage / f"{config}_seed{seed}"


def machine():
    packages = ["pybullet", "panda-gym", "stable-baselines3", "gymnasium", "torch", "numpy"]
    return {"platform": platform.platform(), "processor": platform.machine(), "python": platform.python_version(),
            **{p: metadata.version(p) for p in packages}}


def make_model(config, seed, steps, out, normalize=False):
    """The model exactly as every run trains it: training environments with one shared curriculum, 256x256 network,
    the Part C configuration. Training-environment logs go to `out`. normalize=True (PPO only) wraps the training
    environments in VecNormalize (observations and rewards, VecNormalize defaults, gamma of the configuration)."""
    algo = config.split("_")[0]
    n_envs = N_ENVS[algo]

    def make_train_env(rank):
        def _init():
            env = V.make_env(randomisation=V.TRAIN_RANDOMISATION, curriculum_steps=int(V.CURRICULUM_FRACTION * steps / n_envs), **ENV_FLAGS)
            env = Monitor(env, filename=str(out / f"train_env_{rank}"), info_keywords=("start_type", "is_success"))
            env.reset(seed=seed + 10_000 * rank)              # no environment seed is shared between runs
            env.action_space.seed(seed + 10_000 * rank)
            return env
        return _init

    venv = DummyVecEnv([make_train_env(rank) for rank in range(n_envs)])
    shared_curriculum = V.CurriculumState()                   # one curriculum per policy, whatever the number of envs
    for e in venv.envs:
        e.unwrapped.curriculum = shared_curriculum
    if normalize:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True, gamma=CONFIGS[config]["gamma"])
    kwargs = dict(CONFIGS[config])
    kwargs["policy_kwargs"] = {"net_arch": dict(pi=[256, 256], vf=[256, 256]) if algo == "PPO" else [256, 256]}
    if algo == "TD3":
        sigma, n = kwargs.pop("noise_sigma"), venv.action_space.shape[-1]
        kwargs["action_noise"] = NormalActionNoise(np.zeros(n), sigma * np.ones(n))
    model = ALGOS[algo]("MlpPolicy", venv, seed=seed, device="cpu", verbose=0, **kwargs)
    return model, venv, shared_curriculum


class SaveObsNorm(BaseCallback):
    """Saves the VecNormalize observation statistics next to each checkpoint (same step number), as plain arrays."""

    def __init__(self, save_freq, path):
        super().__init__()
        self.save_freq, self.path = save_freq, path

    def _on_step(self):
        if self.n_calls % self.save_freq == 0:
            save_obs_norm(self.training_env, self.path / f"obs_norm_{self.num_timesteps}_steps.npz")
        return True


def save_obs_norm(venv, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, mean=venv.obs_rms.mean, var=venv.obs_rms.var, clip_obs=venv.clip_obs, epsilon=venv.epsilon)


def load_obs_norm(path):
    d = np.load(path)
    return {k: d[k] for k in ("mean", "var", "clip_obs", "epsilon")}


def train(config, seed, stage, smoke, normalize=False, equal_time=False):
    algo = config.split("_")[0]
    steps = 3_000 if smoke else (STEPS_EQUAL_TIME[algo] if equal_time else STEPS)
    n_envs = N_ENVS[algo]
    out = run_dir(config, seed, stage, smoke, normalize, equal_time)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); set_random_seed(seed)
    model, venv, shared_curriculum = make_model(config, seed, steps, out, normalize)
    every = 1_000 if smoke else 25_000
    save_freq = max(every // n_envs, 1)
    callbacks = [CheckpointCallback(save_freq=save_freq, save_path=str(out / "checkpoints"), name_prefix="model")]
    if normalize:
        callbacks.append(SaveObsNorm(save_freq, out / "checkpoints"))
    t0 = time.perf_counter()
    model.learn(total_timesteps=steps, callback=CallbackList(callbacks))
    minutes = (time.perf_counter() - t0) / 60
    model.save(str(out / "model_final"))
    if normalize:
        save_obs_norm(venv, out / "obs_norm_final.npz")
        model.obs_norm = load_obs_norm(out / "obs_norm_final.npz")
    venv.close()
    print(f"trained {config} seed {seed} ({stage}): {model.num_timesteps:,} steps in {minutes:.1f} min", flush=True)
    summary = {"protocol": PROTOCOL, "smoke test (not a result)": smoke, "config": config, "algorithm": algo,
               "observation and reward normalisation": normalize, "equal-time budget": equal_time,
               "settings": CONFIGS[config],
               "stage": stage, "seed": seed, "steps": int(model.num_timesteps), "training minutes": minutes,
               "test seeds": [STAGES[stage]["test_start"], STAGES[stage]["test_start"] + 49], "machine": machine(),
               "curriculum stages passed": shared_curriculum.log}
    return model, out, summary


def load_model(algo, path):
    """Loads only the network weights and plain settings of a saved model. The pickled parts (spaces, schedules, buffer
    classes) are replaced, so a model saved on another computer with other numpy/Python versions loads the same way,
    and nothing in a received file is unpickled."""
    env = V.make_env(**ENV_FLAGS)
    replace = {"policy_class": ALGOS[algo].policy_aliases["MlpPolicy"], "observation_space": env.observation_space,
               "action_space": env.action_space, "lr_schedule": lambda _: 0.0, "clip_range": 0.2, "train_freq": 1,
               "_last_obs": None, "_last_episode_starts": None, "_last_original_obs": None, "ep_info_buffer": None,
               "ep_success_buffer": None, "rollout_buffer_class": None, "replay_buffer_class": None}
    env.close()
    return ALGOS[algo].load(str(path), device="cpu", custom_objects=replace)


def policy(m):
    """Deterministic actions. A model trained with VecNormalize sees its observations normalised with the statistics
    saved at the end of training (or with its checkpoint), frozen, exactly as VecNormalize.normalize_obs does."""
    norm = getattr(m, "obs_norm", None)
    if norm is None:
        return lambda obs, env: m.predict(obs, deterministic=True)[0]
    scale = np.sqrt(norm["var"] + norm["epsilon"])
    return lambda obs, env: m.predict(np.clip((obs - norm["mean"]) / scale, -norm["clip_obs"], norm["clip_obs"]).astype(np.float32),
                                      deterministic=True)[0]


def summarise(rows):
    df = pd.DataFrame(rows)
    ok = df[df.success]
    return {"success %": 100 * df.success.mean(), "reached %": 100 * df.reached.mean(), "lifted %": 100 * df.lifted.mean(),
            "failure %": 100 * df.failure.mean(), "failed grasp %": 100 * df.failed_grasp.mean(),
            "mean reward": df.reward.mean(), "mean length": df.length.mean(), "regrasps/ep": df.regrasps.mean(),
            "success by item %": {k: 100 * float(g.success.mean()) for k, g in df.groupby("item")},
            "outcomes": df.outcome.value_counts().to_dict(),
            "movement (successful episodes)": {"steps": ok.length.mean(), "path m": ok.path.mean(),
                                               "gripper switches": ok.gripper_switches.mean(), "max lift m": ok.max_lift.mean()}
            if len(ok) else None,
            "movement (all episodes)": {"path m": df.path.mean(), "gripper switches": df.gripper_switches.mean()}}


def trays(act, seeds, dense):
    out = {}
    for n_items in (3, 4):
        results = [V.clear_tray(act, n_items, seed, dense_tray=dense, **ENV_FLAGS) for seed in seeds]
        attempts = [r for t in results for r in t["per_item"]]
        out[f"{n_items} items"] = {"per-item success %": 100 * np.mean([r["outcome"] == "success" for r in attempts]),
                                   "whole-tray success %": 100 * np.mean([t["whole_tray"] for t in results]),
                                   "items sorted per tray": float(np.mean([t["sorted"] for t in results])),
                                   "items attempted per tray": float(np.mean([t["attempted"] for t in results])),
                                   "items placed per tray": float(np.mean([t["items"] for t in results]))}
    return out


def from_starts(m, seeds):
    """Test success from each curriculum start (diagnostic: which part of the task is learned)."""
    out = {}
    for start in V.REVERSE_ORDER + V.APPROACH_STAGES:
        rows = V.run_episodes(policy(m), seeds, randomisation=V.NOMINAL, options={"start_type": start}, **ENV_FLAGS)
        out[start] = 100 * np.mean([r["success"] for r in rows])
    return out


def first_close_errors(m, seeds):
    """Pose of the gripper the first time it closes next to the item, from the normal start."""
    env = V.make_env(**ENV_FLAGS)
    core, rows, act = env.unwrapped, [], policy(m)
    for seed in seeds:
        obs, info = env.reset(seed=seed)
        done, previous, first = False, 1.0, None
        while not done:
            a = act(obs, env)
            task, robot = core.task, core.robot
            lo, hi = task.item_aabb()
            ee = np.asarray(robot.get_ee_position())
            if first is None and a[-1] < 0 <= previous and V.aabb_distance(ee, lo, hi) < 0.03:
                position, yaw, _ = task.grasp_pose()
                first = {"round": abs(task.local_size[0] - task.local_size[1]) < 0.01,
                         "yaw error deg": float(np.degrees(abs(((robot.get_yaw() - yaw + np.pi / 2) % np.pi) - np.pi / 2))),
                         "xy offset cm": float(100 * np.linalg.norm(ee[:2] - position[:2])),
                         "height error cm": float(100 * (ee[2] - position[2]))}
            previous = a[-1]
            obs, r, te, tr, info = env.step(a)
            done = te or tr
        if first:
            rows.append(first)
    env.close()
    df = pd.DataFrame(rows)
    if df.empty:
        return {"episodes with a close next to the item": 0}
    return {"episodes with a close next to the item": len(df),
            "median yaw error deg (non-round items)": float(df[~df["round"]]["yaw error deg"].median()) if (~df["round"]).any() else None,
            "median xy offset cm": float(df["xy offset cm"].median()), "median height error cm": float(df["height error cm"].median())}


def evaluate(model, out, summary, smoke, csv_name="test_nominal.csv"):
    n = 3 if smoke else 50
    start = summary["test seeds"][0]
    test = list(range(start, start + n))
    act = policy(model)
    nominal_rows = V.run_episodes(act, test, randomisation=V.NOMINAL, **ENV_FLAGS)
    pd.DataFrame(nominal_rows).to_csv(out / csv_name, index=False)
    summary["test: nominal"] = summarise(nominal_rows)
    summary["selection: nominal (seeds 1000-1049)"] = summarise(V.run_episodes(act, list(SELECTION_SEEDS)[:n], randomisation=V.NOMINAL, **ENV_FLAGS))
    summary["test: training randomisation"] = summarise(V.run_episodes(act, test, randomisation=V.TRAIN_RANDOMISATION, **ENV_FLAGS))
    for label, rand in V.OUT_OF_RANGE.items():
        summary[f"test: {label}"] = summarise(V.run_episodes(act, test, randomisation=rand, **ENV_FLAGS))
    tray_seeds = test[:1 if smoke else 20]
    summary["tray clearing (20 trays, nominal)"] = trays(act, tray_seeds, dense=False)
    summary["dense tray clearing (20 trays, nominal)"] = trays(act, tray_seeds, dense=True)
    curve = []
    for path in sorted((out / "checkpoints").glob("model_*_steps.zip"), key=lambda p: int(p.stem.split("_")[1])):
        m = load_model(summary["algorithm"], path)
        if summary.get("observation and reward normalisation"):
            m.obs_norm = load_obs_norm(path.parent / f"obs_norm_{path.stem.split('_')[1]}_steps.npz")
        rows = V.run_episodes(policy(m), test, randomisation=V.NOMINAL, **ENV_FLAGS)
        curve.append({"steps": int(path.stem.split("_")[1]), "success %": 100 * np.mean([r["success"] for r in rows]),
                      "lifted %": 100 * np.mean([r["lifted"] for r in rows]), "reached %": 100 * np.mean([r["reached"] for r in rows])})
    summary["checkpoint test success (nominal)"] = curve
    summary["test success from curriculum starts (30 seeds)"] = from_starts(model, test[:min(30, n)])
    summary["first close next to the item (nominal test)"] = first_close_errors(model, test)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=list(CONFIGS))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--stage", choices=list(STAGES))
    parser.add_argument("--smoke", action="store_true", help="3,000 steps and a few episodes: checks that the code runs")
    parser.add_argument("--normalize", action="store_true", help="supplementary comparison: PPO with VecNormalize")
    parser.add_argument("--equal-time", action="store_true", help="supplementary comparison: SAC's training time (PPO 4M, TD3 1.39M steps)")
    parser.add_argument("--evaluate-only", metavar="RUN_DIR", help="evaluate model_final.zip of a finished run again")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.evaluate_only:
        out = Path(args.evaluate_only)
        old = json.load(open(out / "summary.json"))
        model = load_model(old["algorithm"], out / "model_final.zip")
        if old.get("observation and reward normalisation"):
            model.obs_norm = load_obs_norm(out / "obs_norm_final.npz")
        keep = ["protocol", "smoke test (not a result)", "config", "algorithm", "observation and reward normalisation",
                "equal-time budget", "equal-time budget (4M steps)", "settings", "stage", "seed", "steps", "training minutes", "test seeds", "curriculum stages passed"]
        summary = {**{k: old.get(k) for k in keep}, "machine (training)": old["machine"], "machine": machine()}
        summary = evaluate(model, out, summary, old["smoke test (not a result)"], "test_nominal_reevaluated.csv")
        json.dump(summary, open(out / "summary_reevaluated.json", "w"), indent=2, default=float)
        print("saved", out / "summary_reevaluated.json")
        return
    if not (args.config and args.seed is not None and args.stage):
        parser.error("--config, --seed and --stage are required (or --evaluate-only RUN_DIR)")
    if args.seed not in STAGES[args.stage]["seeds"]:
        parser.error(f"the {args.stage} stage uses seeds {STAGES[args.stage]['seeds']}")
    if args.normalize and not args.config.startswith("PPO"):
        parser.error("--normalize is the supplementary PPO comparison; SAC and TD3 run without normalisation")
    if args.equal_time and (args.config.split("_")[0] not in STEPS_EQUAL_TIME or args.stage != "final" or args.normalize):
        parser.error("--equal-time is the supplementary comparison for PPO and TD3: final stage, no normalisation")
    model, out, summary = train(args.config, args.seed, args.stage, args.smoke, args.normalize, args.equal_time)
    summary = evaluate(model, out, summary, args.smoke)
    json.dump(summary, open(out / "summary.json", "w"), indent=2, default=float)
    short = {k: round(v["success %"], 1) for k, v in summary.items() if k.startswith("test:")}
    print(json.dumps({"run": f"{args.stage}/{out.name}", "test success %": short,
                      "trays": summary["tray clearing (20 trays, nominal)"]}, indent=1, default=float))


if __name__ == "__main__":
    main()
