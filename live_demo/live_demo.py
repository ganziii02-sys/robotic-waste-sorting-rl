"""Live demo for the presentation: runs a trained model on a test episode and draws it below the notebook cell as it
happens (presentation/Live_Demo.ipynb).

    demo = LiveDemo("SAC")     # loads the final model (training seed 31) and builds the scene, about 2 s
    demo.run(3000)             # test episode 3000, drawn frame by frame, about 10 s

The policy (deterministic actions), physics and episodes are those of the evaluation; frames are taken between the
physics substeps of each action, as in make_demo_video.py, so the slow motion shows the real motion."""
import io
import sys
import time
from pathlib import Path

from IPython.display import Image, display
from PIL import Image as PILImage

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_demo_video as D  # noqa: E402

SIZE, FPS, EVERY = (960, 540), 10, 5               # 4 frames per action, shown at 10 per second: 0.1x speed


def jpeg(frame):
    buffer = io.BytesIO()
    PILImage.fromarray(frame).save(buffer, format="JPEG", quality=88)
    return Image(data=buffer.getvalue(), format="jpeg")


class LiveDemo:
    def __init__(self, algo="SAC"):
        self.algo = algo
        self.act = D.load(algo)
        self.env = D.W.make_env(**D.DISPLAY)
        self.core = self.env.unwrapped
        self.check = D.expected(algo)
        print(f"Loaded the final {algo} model ({D.RUNS[algo]}) and the scene with the seven YCB items.")

    def frame(self, bottom, badge=None):
        return D.caption(D.shot(self.core.sim, D.CLOSE, SIZE), chip=(self.algo, D.COLORS[self.algo]),
                         title=f"final model, training seed 31  ·  test episode {self.seed}",
                         speed=f"slow motion, about {FPS * EVERY * 0.002:g}×",
                         bottom=bottom, right=f"step {self.core.elapsed_steps}", badge=badge)

    def run(self, seed=3000):
        self.seed = seed
        obs, info = self.env.reset(seed=seed)
        self.env.action_space.seed(seed)
        bottom = f"{D.NAMES[info['item']]}  →  {info['category']} hole"
        handle = display(jpeg(self.frame(bottom)), display_id=True)
        due = time.perf_counter() + 1.0                 # the start for a second, then one frame every 1/FPS s

        def on_frame():
            nonlocal due
            image = jpeg(self.frame(bottom))
            time.sleep(max(0.0, due - time.perf_counter()))
            handle.update(image)
            due = max(due + 1 / FPS, time.perf_counter())
        D.record_substeps(self.env, EVERY, on_frame)
        done = False
        while not done:
            obs, reward, terminated, truncated, info = self.env.step(self.act(obs, self.env))
            done = terminated or truncated
        steps = self.core.elapsed_steps
        if info["is_success"]:
            badge, line = (f"Sorted into the {info['category']} bin  ·  {steps} steps", D.GOOD), \
                f"sorted into the {info['category']} bin in {steps} steps ({steps * 0.04:.2f} s of robot time)"
        else:
            badge, line = (f"Not sorted ({info['outcome'] or 'time limit'})", D.BAD), f"not sorted ({info['outcome'] or 'time limit'})"
        handle.update(jpeg(self.frame(bottom, badge)))
        print(f"Test episode {seed}: {D.NAMES[info['item']]} -> {info['category']}: {line}.")
        if seed in self.check.index:
            same = (info["is_success"], steps) == (bool(self.check.loc[seed, "success"]), int(self.check.loc[seed, "length"]))
            print("Same outcome and length as in the evaluation." if same else "Differs from the evaluation record.")
