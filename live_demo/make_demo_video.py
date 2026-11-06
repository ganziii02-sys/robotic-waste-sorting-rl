"""Demo video of the final agents, rendered from the trained models (1280x720, 25 fps, captions, no sound).

    python live_demo/make_demo_video.py              # renders every part, then joins them into live_demo/demo_video/Group53_3H_demo.mp4
    python live_demo/make_demo_video.py sac join     # re-renders one part and joins again

Each part runs in its own process, so only one display environment (about 1 GB) exists at a time. The episodes are the
final test episodes in order (seed 3000 onwards) with the seed-31 models, and their outcomes are checked against the
evaluation files. Frames are taken between the physics substeps of each action, so the slow motion shows the real
motion, and the policy, physics and outcomes are exactly those of the evaluation."""
import os
import subprocess
import sys
from collections import deque
from functools import lru_cache
from pathlib import Path

import imageio.v2 as imageio
import imageio_ffmpeg
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

PKG = Path(__file__).resolve().parent.parent                          # the Code folder
OUT = Path(__file__).resolve().parent / "demo_video"
sys.path.insert(0, str(PKG))
os.chdir(PKG)
import train_eval as T  # noqa: E402
import waste_env as W  # noqa: E402

WIDTH, HEIGHT, FPS, FADE = 1280, 720, 25, 6
COLORS = {"SAC": "#2a78d6", "TD3": "#eb6834", "PPO": "#1baf7a"}          # as in the report
GOOD, BAD, CARD, INK, MUTED = "#1f8a4c", "#8f3a33", "#16181d", "#ffffff", "#b9bcc4"
RUNS = {"SAC": "SAC_robotics_gamma_seed31", "TD3": "TD3_robotics_gamma_seed31", "PPO": "PPO_small_lr_seed31"}
CAMERA = dict(target_position=np.array([-0.06, 0.0, 0.07]), distance=1.0, yaw=60, pitch=-30)    # whole scene
CLOSE = dict(target_position=np.array([-0.05, 0.0, 0.08]), distance=0.82, yaw=60, pitch=-28)    # episodes
DISPLAY = {**T.ENV_FLAGS, "visual": True}                              # textured meshes; same collision shapes
NAMES = {"006_mustard_bottle": "mustard bottle", "065-a_cups": "cup (065-a)", "065-c_cups": "cup (065-c)",
         "004_sugar_box": "sugar box", "009_gelatin_box": "gelatin box", "005_tomato_soup_can": "tomato soup can",
         "010_potted_meat_can": "potted meat can"}
TEAM = "Team 3H, Project 53  ·  Hyojeong Yu  ·  Heesu Kim  ·  Hyunbeen Suk"


# ---------------------------------------------------------------- drawing
@lru_cache(maxsize=None)
def font(size, bold=False):
    try:
        return ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial" + (" Bold" if bold else "") + ".ttf", int(size))
    except OSError:                                                   # not macOS: Arial on Windows, else Pillow's font
        try:
            return ImageFont.truetype("arialbd.ttf" if bold else "arial.ttf", int(size))
        except OSError:
            return ImageFont.load_default(int(size))


def text_size(d, text, f):
    l, t, r, b = d.textbbox((0, 0), text, font=f)
    return r - l, b - t


def wrap(d, text, f, width):
    lines, line = [], ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if line and text_size(d, trial, f)[0] > width:
            lines.append(line)
            line = word
        else:
            line = trial
    return lines + [line]


def box(d, xy, text, f, fill, ink=INK, pad=(14, 9), anchor="lt"):
    """Text on a rounded box; xy is the box's top left (anchor "lt"), top right ("rt") or bottom centre ("mb")."""
    w, h = text_size(d, text, f)
    bw, bh = w + 2 * pad[0], h + 2 * pad[1] + 4
    x, y = xy
    x = {"lt": x, "rt": x - bw, "mb": x - bw / 2}[anchor]
    y = y - bh if anchor == "mb" else y
    d.rounded_rectangle((x, y, x + bw, y + bh), radius=bh * 0.3, fill=fill)
    d.text((x + pad[0], y + pad[1]), text, font=f, fill=ink)
    return x, y, x + bw, y + bh


def caption(frame, chip=None, title=None, speed=None, bottom=None, right=None, badge=None):
    """Algorithm chip and title (top left), playback speed (top right), bottom bar and a result badge above it."""
    img = Image.fromarray(frame).convert("RGBA")
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    w, h = img.size
    s = max(w / WIDTH, 0.7)                                             # half-size panels keep larger text
    x = 20 * s
    if chip:
        x = box(d, (20 * s, 18 * s), chip[0], font(30 * s, True), chip[1], pad=(16 * s, 9 * s))[2] + 10 * s
    if title:
        box(d, (x, 18 * s), title, font(25 * s), (0, 0, 0, 165), pad=(14 * s, 11 * s))
    if speed:
        box(d, (w - 20 * s, 18 * s), speed, font(25 * s), (0, 0, 0, 165), pad=(14 * s, 11 * s), anchor="rt")
    bar = 62 * s
    if bottom or right:
        d.rectangle((0, h - bar, w, h), fill=(0, 0, 0, 175))
        if bottom:
            d.text((22 * s, h - bar / 2), bottom, font=font(28 * s), fill=INK, anchor="lm")
        if right:
            d.text((w - 22 * s, h - bar / 2), right, font=font(26 * s), fill=MUTED, anchor="rm")
    if badge:
        box(d, (w / 2, h - bar - 16 * s), badge[0], font(30 * s, True), badge[1], pad=(20 * s, 11 * s), anchor="mb")
    return np.asarray(Image.alpha_composite(img, layer).convert("RGB"))


def speed_label(every):
    # one frame every `every` substeps of 2 ms, played at 25 fps
    return f"{FPS * every * 0.002:g}× speed"


def project(sim, point, camera, size):
    pc = sim.physics_client
    view = np.array(pc.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=camera["target_position"],
                                                         distance=camera["distance"], yaw=camera["yaw"],
                                                         pitch=camera["pitch"], roll=0, upAxisIndex=2)).reshape(4, 4).T
    proj = np.array(pc.computeProjectionMatrixFOV(fov=60, aspect=size[0] / size[1], nearVal=0.1, farVal=100.0)).reshape(4, 4).T
    p = proj @ view @ np.append(point, 1.0)
    return (p[0] / p[3] + 1) / 2 * size[0], (1 - p[1] / p[3]) / 2 * size[1]


def shot(sim, camera=CLOSE, size=(WIDTH, HEIGHT)):
    return sim.render(width=size[0], height=size[1], **camera)


# ---------------------------------------------------------------- recording
class Part:
    """One mp4 segment that fades in from and out to black over FADE frames."""

    def __init__(self, name, folder=OUT / "parts"):
        folder.mkdir(parents=True, exist_ok=True)
        self.writer = imageio.get_writer(folder / f"{name}.mp4", fps=FPS, codec="libx264", quality=None,
                                         pixelformat="yuv420p", macro_block_size=16, ffmpeg_params=["-crf", "18"])
        self.count, self.tail = 0, deque()

    def add(self, frame, repeat=1):
        for _ in range(repeat):
            k = (self.count + 1) / (FADE + 1)
            self.tail.append(frame if k >= 1 else (frame * k).astype(np.uint8))
            self.count += 1
            if len(self.tail) > FADE:
                self.writer.append_data(self.tail.popleft())

    def close(self):
        n = len(self.tail)
        for i, frame in enumerate(self.tail):
            self.writer.append_data((frame * ((n - i) / (n + 1))).astype(np.uint8))
        self.writer.close()


def record_substeps(env, every, on_frame):
    """Calls on_frame() after every `every`-th physics substep of an action (the same substeps the evaluation runs)."""
    sim = env.unwrapped.sim

    def step():
        for i in range(sim.n_substeps):
            sim.physics_client.stepSimulation()
            if (i + 1) % every == 0:
                on_frame()
    sim.step = step


def load(algo):
    return T.policy(T.load_model(algo, Path("results/final") / RUNS[algo] / "model_final.zip"))


def expected(algo):
    return pd.read_csv(Path("results/final") / RUNS[algo] / "test_nominal.csv").set_index("seed")


def card(draw_fn):
    img = Image.new("RGB", (WIDTH, HEIGHT), CARD)
    draw_fn(img, ImageDraw.Draw(img))
    return np.asarray(img)


# ---------------------------------------------------------------- parts
def part_title():
    def draw(img, d):
        d.text((90, 120), "43008 Reinforcement Learning  ·  Assignment 3  ·  Demo", font=font(26), fill=MUTED)
        y = 170
        for line in ("Robotic Waste Sorting", "with Deep Reinforcement Learning"):
            d.text((90, y), line, font=font(56, True), fill=INK)
            y += 70
        d.text((90, y + 12), "Comparing PPO, TD3 and SAC", font=font(40, True), fill=INK)
        y += 82
        for line in wrap(d, "Trained to sort real household objects with a simulated Franka Panda arm "
                            "(PyBullet, panda-gym, Stable-Baselines3)", font(30), 1100):
            d.text((90, y), line, font=font(30), fill=MUTED)
            y += 42
        x = 90
        for algo in ("PPO", "TD3", "SAC"):
            x = box(d, (x, y + 40), algo, font(28, True), COLORS[algo], pad=(18, 8))[2] + 14
        d.text((90, 600), TEAM, font=font(26), fill=INK)
    part = Part("1_title")
    part.add(card(draw), repeat=int(4.5 * FPS))
    part.close()


def part_task():
    """The scene with labels, while the camera turns slowly around it."""
    env = W.make_env(**DISPLAY)
    obs, info = env.reset(seed=3000)
    core, sim = env.unwrapped, env.unwrapped.sim
    item = core.task.get_achieved_goal()
    ee = np.asarray(core.robot.get_ee_position())
    tray = [np.array([x, y, 0.0]) for x, y in ((W.TRAY_X[0], W.TRAY_Y[0]), (W.TRAY_X[1], W.TRAY_Y[0]),
                                               (W.TRAY_X[1], W.TRAY_Y[1]), (W.TRAY_X[0], W.TRAY_Y[1]))]
    part = Part("2_task")
    n = int(7 * FPS)
    for k in range(n):
        camera = {**CAMERA, "yaw": 40 + 20 * k / (n - 1)}
        img = Image.fromarray(shot(sim, camera)).convert("RGBA")
        layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        p = lambda q: project(sim, q, camera, (WIDTH, HEIGHT))
        d.polygon([p(c) for c in tray], outline=(60, 64, 72, 230), width=3)
        labels = [(p(np.array([W.TRAY_X[0], W.TRAY_Y[0], 0.0])), "tray area (items start here)", (-30, 40), "rt"),
                  (p(item), f"item: {NAMES[info['item']]}", (-60, -80), "mb"),
                  (p(ee), "gripper", (110, -60), "mb")]
        for (x, y), text, (dx, dy), anchor in labels:
            d.line((x, y, x + dx, y + dy), fill=(25, 27, 32, 230), width=3)
            d.ellipse((x - 6, y - 6, x + 6, y + 6), fill=(25, 27, 32, 255), outline=INK, width=2)
            box(d, (x + dx, y + dy), text, font(25, True), (25, 27, 32, 225), anchor=anchor)
        for b, name in zip(W.BIN_POSITIONS, W.CATEGORY_NAMES):          # on the holes themselves
            x, y = p(b)
            box(d, (x, y + 20), name, font(25, True), (25, 27, 32, 225), anchor="mb")
        frame = np.asarray(Image.alpha_composite(img, layer).convert("RGB"))
        part.add(caption(frame, title="The task", bottom="Pick up the item, lift it and drop it through the hole of its category",
                         right="one item, at most 100 actions (4 s)"))
    part.close()
    env.close()


def part_items():
    env = W.make_env(**DISPLAY)
    core, sim = env.unwrapped, env.unwrapped.sim
    shots = []
    for i, (name, category, _) in enumerate(W.ITEMS):
        core.task.force_item = i
        env.reset(seed=3000)
        target = core.task.get_achieved_goal() + np.array([0.0, 0.0, 0.03])
        camera = dict(target_position=target, distance=0.32, yaw=60, pitch=-20)
        shots.append((shot(sim, camera, (290, 210)), name, category))
    env.close()

    def draw(img, d):
        d.text((40, 36), "Seven YCB objects: 3D scans of real products, with their real masses",
               font=font(32, True), fill=INK)
        d.text((40, 84), "The gripper opens 8 cm at most, so the objects were chosen to fit it.", font=font(25), fill=MUTED)
        for k, (shot_, name, category) in enumerate(shots):
            row, col = divmod(k, 4)
            x = 40 + col * 305 + (152 if row else 0)
            y = 150 + row * 280
            img.paste(Image.fromarray(shot_), (x, y))
            d.text((x, y + 222), NAMES[name], font=font(25, True), fill=INK)
            d.text((x + 290, y + 222), category, font=font(25), fill=MUTED, anchor="ra")
    part = Part("3_items")
    part.add(card(draw), repeat=int(5 * FPS))
    part.close()


def part_sac():
    """Test episodes 3000-3004 in order, in slow motion."""
    every, seeds = 5, range(3000, 3005)
    env = W.make_env(**DISPLAY)
    core, sim = env.unwrapped, env.unwrapped.sim
    act, check = load("SAC"), expected("SAC")
    part, now = Part("4_sac"), {}
    title = "First of the five final runs  ·  test episodes 3000–3004 in order, not selected"

    def frame(badge=None):
        return caption(now["raw"], chip=("SAC", COLORS["SAC"]), title=title, speed=speed_label(every), badge=badge,
                       bottom=f"Episode {now['seed']}:  {NAMES[now['item']]}  →  {now['category']} hole",
                       right=f"step {core.elapsed_steps}")

    def on_frame():
        now["raw"] = shot(sim)
        part.add(frame())

    record_substeps(env, every, on_frame)
    for seed in seeds:
        obs, info = env.reset(seed=seed)
        env.action_space.seed(seed)
        now.update(seed=seed, item=info["item"], category=info["category"], raw=shot(sim))
        part.add(frame(), repeat=12)
        done = False
        while not done:
            obs, reward, terminated, truncated, info = env.step(act(obs, env))
            done = terminated or truncated
        steps = core.elapsed_steps
        assert (info["is_success"], steps) == (bool(check.loc[seed, "success"]), int(check.loc[seed, "length"])), seed
        badge = (f"Sorted into the {info['category']} bin  ·  {steps} steps ({steps * 0.04:.2f} s)", GOOD) \
            if info["is_success"] else (f"Not sorted ({info['outcome'] or 'time limit'})", BAD)
        part.add(frame(badge), repeat=int(1.4 * FPS))
        print(seed, info["item"], info["outcome"], steps, flush=True)
    part.close()
    env.close()


def part_tray():
    """The single-item policy applied to every item of test tray 3000, outermost item first (as clear_tray does)."""
    every, seed = 10, 3000
    act = load("SAC")
    reference = W.clear_tray(act, 3, seed, **T.ENV_FLAGS)                   # the evaluation's outcome for this tray
    env = W.make_env(n_distractors=2, **DISPLAY)
    core, sim = env.unwrapped, env.unwrapped.sim
    part, now = Part("5_tray"), {}
    title = "The same policy on a tray of three items  ·  test tray 3000"

    def frame(badge=None, bottom=None):
        return caption(now["raw"], chip=("SAC", COLORS["SAC"]), title=title, speed=speed_label(every), badge=badge,
                       bottom=bottom or f"Item {now['k']} of 3:  {NAMES[now['item']]}  →  {now['category']} hole",
                       right=f"step {core.elapsed_steps}")

    def on_frame():
        now["raw"] = shot(sim)
        part.add(frame())

    record_substeps(env, every, on_frame)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    remaining, results = list(core.task.active), []
    while remaining:
        remaining = [b for b in remaining if core.task.on_table(b)]
        if not remaining:
            break
        target = W.outermost_first(core.task, remaining)
        obs = env.observation(core.retarget(target))
        now.update(k=len(results) + 1, item=core.task.item_name, category=W.CATEGORY_NAMES[core.task.category],
                   raw=shot(sim))
        part.add(frame(), repeat=12)
        done = False
        while not done:
            obs, reward, terminated, truncated, info = env.step(act(obs, env))
            done = terminated or truncated
        results.append({"item": info["item"], "category": info["category"], "outcome": info["outcome"] or "timeout"})
        badge = (f"Sorted into the {info['category']} bin", GOOD) if info["is_success"] else (f"Not sorted ({results[-1]['outcome']})", BAD)
        part.add(frame(badge), repeat=int(1.0 * FPS))
        remaining.remove(target)
    assert results == reference["per_item"], (results, reference["per_item"])
    sorted_ok = sum(r["outcome"] == "success" for r in results)
    part.add(frame((f"{sorted_ok} of 3 items sorted", GOOD if sorted_ok == 3 else BAD),
                   bottom="20 test trays: 60% fully cleared in this run, 52% on average over the five runs"), repeat=int(3 * FPS))
    print(results, flush=True)
    part.close()
    env.close()


def part_compare():
    """SAC, TD3 and PPO on the same test episode, side by side."""
    every, seed, size = 10, 3000, (WIDTH // 2, HEIGHT // 2)
    runs = {}
    for algo in ("SAC", "TD3", "PPO"):
        env = W.make_env(**DISPLAY)
        core, sim = env.unwrapped, env.unwrapped.sim
        act, check = load(algo), expected(algo)
        frames = []
        record_substeps(env, every, lambda: frames.append((shot(sim, size=size), core.elapsed_steps)))
        obs, info = env.reset(seed=seed)
        env.action_space.seed(seed)
        frames.append((shot(sim, size=size), 0))
        done = False
        while not done:
            obs, reward, terminated, truncated, info = env.step(act(obs, env))
            done = terminated or truncated
        assert (info["is_success"], core.elapsed_steps) == (bool(check.loc[seed, "success"]), int(check.loc[seed, "length"])), algo
        runs[algo] = (frames, info, core.elapsed_steps)
        print(algo, info["item"], info["outcome"], core.elapsed_steps, flush=True)
        env.close()

    item, category = runs["SAC"][1]["item"], runs["SAC"][1]["category"]

    def text_panel(d, x0, y0):
        d.rectangle((x0, y0, x0 + size[0], y0 + size[1]), fill=CARD)
        y = y0 + 30
        for line in wrap(d, f"Same test episode ({seed}) and start state for all three: {NAMES[item]} → {category} hole",
                         font(25, True), size[0] - 60):
            d.text((x0 + 30, y), line, font=font(25, True), fill=INK)
            y += 34
        y += 14
        for line in wrap(d, f"{speed_label(every)}. Only SAC lifts the item and sorts it. TD3 and PPO do not lift it "
                            "within the 100-step limit.",
                         font(23), size[0] - 60):
            d.text((x0 + 30, y), line, font=font(23), fill=MUTED)
            y += 31
        y += 18
        d.text((x0 + 30, y), "Test success, 5 runs × 50 episodes", font=font(22), fill=MUTED)
        y += 36
        for algo, value in (("SAC", 89.2), ("TD3", 0.0), ("PPO", 0.0)):
            d.text((x0 + 30, y), algo, font=font(22, True), fill=INK)
            d.rounded_rectangle((x0 + 100, y + 3, x0 + 100 + max(4, 3.6 * value), y + 23), radius=4, fill=COLORS[algo])
            d.text((x0 + 112 + 3.6 * value, y), f"{value:.1f}%", font=font(22), fill=INK)
            y += 32

    part = Part("6_compare")
    n = max(len(r[0]) for r in runs.values())
    for t in list(range(n)) + [n - 1] * int(2.5 * FPS):
        img = Image.new("RGB", (WIDTH, HEIGHT), CARD)
        for k, algo in enumerate(("SAC", "TD3", "PPO")):
            frames, info, steps = runs[algo]
            raw, step = frames[min(t, len(frames) - 1)]
            badge = None
            if t >= len(frames) - 1:
                badge = (f"Sorted · {steps} steps", GOOD) if info["is_success"] else ("Not sorted in 100 steps", BAD)
            panel = caption(raw, chip=(algo, COLORS[algo]), right=f"step {step}", badge=badge)
            img.paste(Image.fromarray(panel), ((k % 2) * size[0], (k // 2) * size[1]))
        text_panel(ImageDraw.Draw(img), size[0], size[1])
        part.add(np.asarray(img))
    part.close()


def part_end():
    def draw(img, d):
        d.text((90, 70), "Results of the final runs (5 training seeds × 50 test episodes each)", font=font(34, True), fill=INK)
        y = 150
        for algo, value in (("SAC", 89.2), ("TD3", 0.0), ("PPO", 0.0)):
            d.text((90, y), algo, font=font(32, True), fill=INK)
            d.rounded_rectangle((190, y + 4, 190 + max(6, 8 * value), y + 36), radius=6, fill=COLORS[algo])
            d.text((210 + 8 * value, y + 2), f"{value:.1f}% success", font=font(30), fill=INK)
            y += 62
        y += 20
        for text in ["SAC was the only algorithm that learned the task, at equal steps (1M) and at approximately equal "
                     "training time (TD3 1.39M steps, PPO 4M steps). PPO with observation and reward normalisation also "
                     "stayed at 0%.",
                     "On a tray of three items, SAC sorted 83.6% of the items and cleared 52% of the trays completely, "
                     "although it was trained on single items only."]:
            for line in wrap(d, text, font(27), 1100):
                d.text((90, y), line, font=font(27), fill=MUTED)
                y += 38
            y += 16
        d.text((90, 640), TEAM, font=font(26), fill=INK)
    part = Part("7_end")
    part.add(card(draw), repeat=int(7 * FPS))
    part.close()


PARTS = {"title": part_title, "task": part_task, "items": part_items, "sac": part_sac, "tray": part_tray,
         "compare": part_compare, "end": part_end}


def join():
    files = sorted((OUT / "parts").glob("*.mp4"))
    listing = OUT / "parts" / "list.txt"
    listing.write_text("".join(f"file '{f.name}'\n" for f in files))
    target = OUT / "Group53_3H_demo.mp4"
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i",
                    str(listing), "-c", "copy", "-movflags", "+faststart", str(target)], check=True)
    print("wrote", target)


if __name__ == "__main__":
    names = sys.argv[1:] or list(PARTS) + ["join"]
    for name in names:
        if name == "join":
            join()
        elif len(names) > 1:
            subprocess.run([sys.executable, __file__, name], check=True)           # one environment per process
        else:
            PARTS[name]()
