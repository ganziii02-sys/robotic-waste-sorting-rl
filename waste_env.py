"""Final-stage environment for team 3H (Group 53): sorting real YCB waste items with a Panda arm (PyBullet).

This is the environment every algorithm uses in the final runs. Do not change this file; train_eval.py fixes every
setting of the final experiment. Same code as the environment used in development (tools/v9_env.py), with the final
settings as defaults (relative observation on, display meshes off) and the asset folder next to this file.

Task: pick the item on the tray, lift it, carry it and let it fall through the hole above the bin of its category
(plastic / paper / metal). Action = (dx, dy, dz, dyaw, gripper) in [-1, 1]. Episode = at most 100 steps.
- reward: potential-based shaping (reach, grasp, lift, carry), -0.01 per step, +5 success, -5 failure;
- regrasp: a lifted item put back on the table is not a failure; it has to be lifted again;
- reverse start-state curriculum (training only): above_bin -> in_hand -> pre_grasp;
- physical randomisation (training): mass and friction x0.7-1.3, size x0.9-1.1; evaluation also outside these ranges;
- evaluation helpers: run_episodes (one item), clear_tray (several items, optionally a dense tray).
"""
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import pybullet
from panda_gym.envs.core import RobotTaskEnv, Task
from panda_gym.envs.robots.panda import Panda
from panda_gym.pybullet import PyBullet

ASSET_DIR = Path(__file__).resolve().parent / "assets" / "ycb"
CATEGORY_NAMES = ("plastic", "paper", "metal")
BIN_POSITIONS = np.array([[0.12, -0.22, 0.0], [0.12, 0.00, 0.0], [0.12, 0.22, 0.0]])
HOLE_HALF = 0.07                 # 14 cm square hole above each bin, no rim
LIFT_CLEARANCE = 0.06            # "lifted" = centre raised 6 cm above its resting height
PROGRESS_SCALE, STEP_COST = 10.0, 0.01
SUCCESS_BONUS, FAIL_PENALTY = 5.0, 5.0
LIFT_WEIGHT, UNLIFTED_BIN_DISTANCE = 5.0, 0.5
MAX_YAW_STEP = 0.25              # rad per step
MAX_EPISODE_STEPS = 100
TRAY_X, TRAY_Y = (-0.24, -0.04), (-0.18, 0.18)   # item centres are placed inside this tray
TRAY_CENTRE = np.array([np.mean(TRAY_X), np.mean(TRAY_Y)])

# Reverse start-state curriculum (training only), same schedule and rule as Part C v8
CURRICULUM_P0, CURRICULUM_FRACTION, EASY_LIFT = 0.5, 0.6, 0.10
REVERSE_ORDER = ("above_bin", "in_hand", "pre_grasp")
REVERSE_WINDOW, REVERSE_THRESHOLD, REVERSE_REPLAY = 20, 0.5, 0.2
# Curriculum revision (approach_stages=True): two more stages behind pre_grasp, so the approach is practised too, and a
# lower threshold to move on, since a stochastic training policy rarely reaches 50% on real items.
APPROACH_STAGES, APPROACH_HEIGHT, APPROACH_THRESHOLD = ("above_item", "above_item_turn"), 0.07, 0.3

# Grasp reward: both finger pads within 2 mm of the item with a mostly horizontal contact normal, gripper not wide open
GRASP_POTENTIAL, CONTACT_DISTANCE, MAX_NORMAL_Z, GRASP_WIDTH = 0.05, 0.002, 0.7, (0.003, 0.075)

# Physical randomisation: (low, high) multipliers for mass and friction, and the item sizes available
TRAIN_RANDOMISATION = {"mass": (0.7, 1.3), "friction": (0.7, 1.3), "scale": (0.9, 0.95, 1.0, 1.05, 1.1)}
NOMINAL = {"mass": (1.0, 1.0), "friction": (1.0, 1.0), "scale": (1.0,)}
OUT_OF_RANGE = {
    "heavy (mass x2)": {"mass": (2.0, 2.0), "friction": (1.0, 1.0), "scale": (1.0,)},
    "slippery (friction x0.5)": {"mass": (1.0, 1.0), "friction": (0.5, 0.5), "scale": (1.0,)},
    "large (size x1.2)": {"mass": (1.0, 1.0), "friction": (1.0, 1.0), "scale": (1.2,)},
    "small (size x0.8)": {"mass": (1.0, 1.0), "friction": (1.0, 1.0), "scale": (0.8,)},
}

# name, category, resting orientation (roll, pitch, yaw) applied before a random yaw
ITEMS = [
    ("006_mustard_bottle", "plastic", (0, 0, 0)), ("065-a_cups", "plastic", (0, 0, 0)), ("065-c_cups", "plastic", (0, 0, 0)),
    ("004_sugar_box", "paper", (0, 0, 0)), ("009_gelatin_box", "paper", (np.pi / 2, 0, 0)),
    ("005_tomato_soup_can", "metal", (0, 0, 0)), ("010_potted_meat_can", "metal", (0, 0, 0)),
]


class CurriculumState:
    """Stage of the reverse curriculum. Training environments that serve one policy (PPO's vectorised environments)
    share one instance, so stage decisions use all of that policy's episodes, as with a single environment."""

    def __init__(self):
        self.frontier, self.history, self.log, self.steps = 0, [], [], 0


class PandaYaw(Panda):
    """Panda whose end effector also rotates about the vertical axis: action = (dx, dy, dz, dyaw, gripper)."""

    def __init__(self, sim, base_position):
        super().__init__(sim, block_gripper=False, base_position=base_position, control_type="ee")
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(5,), dtype=np.float32)
        self.target_yaw = 0.0

    def _orientation(self, yaw):
        down = np.array([1.0, 0.0, 0.0, 0.0])                       # panda-gym's gripper-down orientation
        spin = self.sim.physics_client.getQuaternionFromEuler([0.0, 0.0, yaw])
        return np.array(self.sim.physics_client.multiplyTransforms([0, 0, 0], spin, [0, 0, 0], down)[1])

    def set_action(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        target = self.get_ee_position() + action[:3] * 0.05
        target[2] = max(0.0, target[2])
        self.target_yaw = float(np.clip(self.target_yaw + action[3] * MAX_YAW_STEP, -np.pi / 2, np.pi / 2))
        arm = self.inverse_kinematics(link=self.ee_link, position=target, orientation=self._orientation(self.target_yaw))[:7]
        fingers = self.get_fingers_width() + action[4] * 0.2
        self.control_joints(target_angles=np.concatenate([arm, [fingers / 2, fingers / 2]]))

    def get_yaw(self):
        quat = self.sim.get_link_orientation(self.body_name, self.ee_link)
        axis = np.array(self.sim.physics_client.getMatrixFromQuaternion(quat)).reshape(3, 3) @ np.array([0.0, 1.0, 0.0])
        yaw = np.arctan2(-axis[0], axis[1])
        return (yaw + np.pi / 2) % np.pi - np.pi / 2                 # a parallel gripper is symmetric under 180 degrees

    def get_obs(self):
        yaw = self.get_yaw()
        return np.concatenate([super().get_obs(), [np.sin(yaw), np.cos(yaw)]])

    def reset(self):
        super().reset()
        self.target_yaw = 0.0


def aabb_distance(point, lo, hi):
    return float(np.linalg.norm(np.maximum(0.0, np.maximum(lo - point, point - hi))))


def body_name(item_name, scale):
    return f"{item_name}@{scale:.2f}"


class RealWasteTask(Task):
    def __init__(self, sim, robot, randomisation, items=ITEMS, n_distractors=0, grasp_reward=True, relative_obs=False,
                 visual=True, dense_tray=False):
        super().__init__(sim)
        self.relative_obs, self.visual, self.dense_tray, self.tray_gaps = relative_obs, visual, dense_tray, []
        self.robot, self.items, self.n_distractors = robot, list(items), n_distractors
        self.randomisation, self.grasp_reward = randomisation, grasp_reward
        self.catalogue = json.load(open(ASSET_DIR / "catalogue.json"))
        self.active, self.distractor_start, self.properties = [], {}, {}
        self.item_index, self.category, self.force_item = 0, 0, None
        self.target_body = body_name(self.items[0][0], randomisation["scale"][0])
        self.one_hot = np.zeros(3, dtype=np.float32)
        self.has_been_lifted, self.gripper_open_command = False, True
        with self.sim.no_rendering():
            self._create_scene()

    # ---------------------------------------------------------------- scene
    def _create_scene(self):
        self.sim.physics_client.setPhysicsEngineParameter(deterministicOverlappingPairs=1)
        self.sim.create_plane(z_offset=-0.4)
        x0, x1 = 0.12 - HOLE_HALF, 0.12 + HOLE_HALF
        ys = [-0.35] + [v for b in BIN_POSITIONS for v in (b[1] - HOLE_HALF, b[1] + HOLE_HALF)] + [0.35]
        pieces = [(-0.85, x0, -0.35, 0.35), (x1, 0.25, -0.35, 0.35)] + [(x0, x1, ys[k], ys[k + 1]) for k in range(0, len(ys), 2)]
        for k, (a0, a1, b0, b1) in enumerate(pieces):
            self.sim.create_box(body_name=f"table_{k}", half_extents=np.array([(a1 - a0) / 2, (b1 - b0) / 2, 0.2]), mass=0.0,
                                position=np.array([(a0 + a1) / 2, (b0 + b1) / 2, -0.2]), rgba_color=np.array([0.95, 0.95, 0.95, 1.0]))
        self.parking, self.base_friction = {}, {}
        for i, (name, _, _) in enumerate(self.items):
            for j, scale in enumerate(self.randomisation["scale"]):
                body = body_name(name, scale)
                k = len(self.parking)
                self.parking[body] = np.array([1.5 + 0.4 * (k % 6), -1.2 + 0.4 * (k // 6), -0.25])
                # visual=False skips the textured display meshes (about 2 GB per environment with all scaled copies);
                # collision shapes, and therefore the physics, are identical
                uid = self.sim.physics_client.loadURDF(str(ASSET_DIR / name / "model.urdf"), basePosition=self.parking[body],
                                                       globalScaling=float(scale), useFixedBase=False,
                                                       flags=0 if self.visual else pybullet.URDF_IGNORE_VISUAL_SHAPES)
                self.sim._bodies_idx[body] = uid
                self.base_friction[body] = self.sim.physics_client.getDynamicsInfo(uid, -1)[1]
                self.sim.physics_client.changeDynamics(uid, -1, mass=0.0)   # parked items are static

    def _uid(self, body=None):
        return self.sim._bodies_idx[body or self.target_body]

    @property
    def item_name(self):
        return self.items[self.item_index][0]

    def item_aabb(self, body=None):
        lo, hi = self.sim.physics_client.getAABB(self._uid(body))
        return np.array(lo), np.array(hi)

    def _size(self, index, scale):
        size = np.array(self.catalogue[self.items[index][0]]["size_m"], dtype=np.float32) * scale
        return size[[0, 2, 1]] if self.items[index][2][0] else size        # stood on its edge: y and z swap

    def _place(self, body, rpy, xy):
        pc, uid = self.sim.physics_client, self._uid(body)
        orientation = pc.getQuaternionFromEuler(rpy)
        pc.resetBasePositionAndOrientation(uid, [xy[0], xy[1], 0.5], orientation)
        lo, _ = pc.getAABB(uid)
        centre = np.array(pc.getBasePositionAndOrientation(uid)[0])
        pc.resetBasePositionAndOrientation(uid, [xy[0], xy[1], centre[2] - lo[2] + 0.002], orientation)
        pc.resetBaseVelocity(uid, [0, 0, 0], [0, 0, 0])

    # ---------------------------------------------------------------- episode
    def _draw(self, index):
        r = self.randomisation
        scale = float(r["scale"][int(self.np_random.integers(len(r["scale"])))])
        return {"index": index, "scale": scale, "mass_scale": float(self.np_random.uniform(*r["mass"])),
                "friction_scale": float(self.np_random.uniform(*r["friction"])), "body": body_name(self.items[index][0], scale)}

    def _radius(self, p):
        size = self._size(p["index"], p["scale"])
        return 0.5 * float(np.hypot(size[0], size[1]))                   # footprint radius for any yaw

    def _activate(self, p, xy):
        pc, name = self.sim.physics_client, self.items[p["index"]][0]
        roll, pitch, yaw = self.items[p["index"]][2]
        pc.changeDynamics(self._uid(p["body"]), -1, mass=self.catalogue[name]["mass_kg"] * p["mass_scale"],
                          lateralFriction=self.base_friction[p["body"]] * p["friction_scale"])
        self._place(p["body"], (roll, pitch, yaw + self.np_random.uniform(-np.pi, np.pi)), xy)
        self.properties[p["body"]] = p

    def _free_spot(self, radius, placed):
        for _ in range(500):
            xy = np.array([self.np_random.uniform(*TRAY_X), self.np_random.uniform(*TRAY_Y)])
            if all(np.linalg.norm(xy - q) > radius + r + 0.02 for q, r in placed):
                return xy
        return None

    def _dense_place(self, p):
        """Dense tray (evaluation): put an extra item where it touches or nearly touches (gap <= 1 cm) an item already on
        the tray without overlapping it. Returns the gap in metres, or None if no such spot was found."""
        pc, name = self.sim.physics_client, self.items[p["index"]][0]
        roll, pitch, yaw = self.items[p["index"]][2]
        uid, others = self._uid(p["body"]), [self._uid(b) for b in self.active]
        pc.changeDynamics(uid, -1, mass=self.catalogue[name]["mass_kg"] * p["mass_scale"],
                          lateralFriction=self.base_friction[p["body"]] * p["friction_scale"])
        for _ in range(600):
            xy = np.array([self.np_random.uniform(*TRAY_X), self.np_random.uniform(*TRAY_Y)])
            self._place(p["body"], (roll, pitch, yaw + self.np_random.uniform(-np.pi, np.pi)), xy)
            gaps = [min((c[8] for c in pc.getClosestPoints(uid, o, 0.05)), default=np.inf) for o in others]
            if min(gaps) >= -0.001 and min(gaps) <= 0.01:
                self.properties[p["body"]] = p
                return float(min(gaps))
        pc.resetBasePositionAndOrientation(uid, self.parking[p["body"]], [0, 0, 0, 1])
        pc.changeDynamics(uid, -1, mass=0.0)
        return None

    def reset(self):
        pc = self.sim.physics_client
        self.tray_gaps = []
        for body in self.active:                                          # park everything from the last episode
            uid = self._uid(body)
            pc.resetBasePositionAndOrientation(uid, self.parking[body], [0, 0, 0, 1])
            pc.resetBaseVelocity(uid, [0, 0, 0], [0, 0, 0])
            pc.changeDynamics(uid, -1, mass=0.0)
        category = int(self.np_random.integers(0, 3))
        choices = [i for i, item in enumerate(self.items) if item[1] == CATEGORY_NAMES[category]]
        index = int(choices[self.np_random.integers(0, len(choices))])
        if self.force_item is not None:                                   # testing: always this item
            index = self.force_item
        target = self._draw(index)
        placed = [(self._free_spot(self._radius(target), []), self._radius(target))]
        self._activate(target, placed[0][0])
        self.active, self.distractor_start = [target["body"]], {}
        others = [i for i, item in enumerate(self.items) if item[1] != self.items[index][1]]
        for other in self.np_random.permutation(others)[:self.n_distractors]:   # multi-item tray (evaluation only)
            p = self._draw(int(other))
            if p["body"] in self.active:
                continue
            if self.dense_tray:
                gap = self._dense_place(p)
                if gap is None:
                    continue
                self.tray_gaps.append(gap)
                self.active.append(p["body"])
                self.distractor_start[p["body"]] = np.array(pc.getBasePositionAndOrientation(self._uid(p["body"]))[0])
                continue
            xy = self._free_spot(self._radius(p), placed)
            if xy is None:
                continue
            self._activate(p, xy)
            placed.append((xy, self._radius(p)))
            self.active.append(p["body"])
            self.distractor_start[p["body"]] = np.array(pc.getBasePositionAndOrientation(self._uid(p["body"]))[0])
        self.set_target(target["body"])

    def set_target(self, body):
        """Makes `body` the item to sort (at reset and when clearing a tray)."""
        p = self.properties[body]
        self.target_body, self.item_index = body, p["index"]
        self.category = CATEGORY_NAMES.index(self.items[self.item_index][1])
        self.one_hot = np.eye(3, dtype=np.float32)[self.category]
        self.goal = BIN_POSITIONS[self.category].copy()
        self.rest_height = float(self.get_achieved_goal()[2])
        self.local_size = self._size(self.item_index, p["scale"])
        self.has_been_lifted, self.gripper_open_command = False, True
        self.previous_potential = self.distance_potential()

    def grasp_pose(self):
        """Gripper position, yaw and opening that hold the target across its narrow side."""
        pc = self.sim.physics_client
        lo, hi = self.item_aabb()
        centre = (lo + hi) / 2
        grasp_z = max(hi[2] - 0.03, centre[2]) if hi[2] - lo[2] > 0.08 else centre[2]
        yaw_item = pc.getEulerFromQuaternion(pc.getBasePositionAndOrientation(self._uid())[1])[2]
        narrow = yaw_item + (0.0 if self.local_size[0] < self.local_size[1] else np.pi / 2)
        return np.array([centre[0], centre[1], grasp_z]), float(narrow % np.pi - np.pi / 2), float(min(self.local_size[:2]))

    def _set_arm(self, position, yaw, finger_width):
        for _ in range(5):                                               # iterate the IK from the current configuration
            angles = self.robot.inverse_kinematics(link=self.robot.ee_link, position=position,
                                                   orientation=self.robot._orientation(yaw))
            self.robot.set_joint_angles(np.concatenate([angles[:7], [finger_width / 2, finger_width / 2]]))
        self.robot.target_yaw = yaw

    def apply_start(self, start_type):
        """Curriculum starts. pre_grasp: gripper open around the item at grasp height. in_hand: item held EASY_LIFT above
        its tray position. above_bin: item held with its centre EASY_LIFT higher, above the centre of its correct hole."""
        if start_type == "normal":
            return
        position, yaw, width = self.grasp_pose()
        if start_type == "pre_grasp":
            self._set_arm(position, yaw, 0.08)
        elif start_type == "above_item":                                   # open, wrist aligned, 7 cm above the grasp point
            self._set_arm(position + [0, 0, APPROACH_HEIGHT], yaw, 0.08)
        elif start_type == "above_item_turn":                              # same, wrist at a random angle
            self._set_arm(position + [0, 0, APPROACH_HEIGHT], float(self.np_random.uniform(-np.pi / 2, np.pi / 2)), 0.08)
        else:
            pc, uid = self.sim.physics_client, self._uid()
            base, orientation = pc.getBasePositionAndOrientation(uid)
            lo, hi = self.item_aabb()
            shift = np.array([0.0, 0.0, EASY_LIFT])
            if start_type == "above_bin":
                shift[:2] = self.goal[:2] - (lo + hi)[:2] / 2
            pc.resetBasePositionAndOrientation(uid, np.array(base) + shift, orientation)
            pc.resetBaseVelocity(uid, [0, 0, 0], [0, 0, 0])
            self._set_arm(position + shift, yaw, max(width - 0.004, 0.0))
        self.get_obs()                                                   # updates the lift flag
        self.previous_potential = self.distance_potential()

    def on_table(self, body):
        lo, hi = self.item_aabb(body)
        centre = (lo + hi) / 2
        return lo[2] > -0.02 and TRAY_X[0] - 0.2 < centre[0] < 0.25 and abs(centre[1]) < 0.35

    def grasped(self):
        robot, pc = self.robot, self.sim.physics_client
        panda, item = self.sim._bodies_idx[robot.body_name], self._uid()
        width = robot.get_fingers_width()
        if not GRASP_WIDTH[0] <= width <= GRASP_WIDTH[1]:
            return False
        for finger in robot.fingers_indices:
            points = pc.getClosestPoints(panda, item, CONTACT_DISTANCE, linkIndexA=int(finger), linkIndexB=-1)
            if not any(abs(p[7][2]) < MAX_NORMAL_Z for p in points):     # contactNormalOnB: a side, not the top
                return False
        return True

    def get_obs(self):
        if self.get_achieved_goal()[2] >= self.rest_height + LIFT_CLEARANCE:
            self.has_been_lifted = True
        uid = self._uid()
        pc = self.sim.physics_client
        return np.concatenate([
            self.get_achieved_goal(),
            pc.getEulerFromQuaternion(pc.getBasePositionAndOrientation(uid)[1]),
            pc.getBaseVelocity(uid)[0],
            pc.getBaseVelocity(uid)[1],
            self.local_size,                                  # object size (what a camera bounding box would give)
            self.one_hot,
            [float(self.has_been_lifted)],
            self.relative_features() if self.relative_obs else [],
        ]).astype(np.float32)

    def relative_features(self):
        """Same information as the item pose, in a form easier to learn from: the vector from the gripper to the grasp
        point, and the wrist angle relative to the item's narrow side as (sin 2d, cos 2d), smooth under the gripper's
        180-degree symmetry."""
        position, yaw, _ = self.grasp_pose()
        d = self.robot.get_yaw() - yaw
        return np.concatenate([position - np.asarray(self.robot.get_ee_position()), [np.sin(2 * d), np.cos(2 * d)]])

    def get_achieved_goal(self):
        return np.array(self.sim.physics_client.getBasePositionAndOrientation(self._uid())[0], dtype=np.float32)

    def distance_potential(self):
        lo, hi = self.item_aabb()
        reach = aabb_distance(np.asarray(self.robot.get_ee_position()), lo, hi)
        item = self.get_achieved_goal()
        if self.has_been_lifted:
            bin_distance, lift, grasp = np.linalg.norm(item[:2] - self.goal[:2]), LIFT_CLEARANCE, 1.0
        else:
            bin_distance = UNLIFTED_BIN_DISTANCE
            lift = float(np.clip(item[2] - self.rest_height, 0.0, LIFT_CLEARANCE))
            grasp = float(self.grasped()) if self.grasp_reward else 0.0
        return -float(reach + bin_distance) + LIFT_WEIGHT * lift + (GRASP_POTENTIAL * grasp if self.grasp_reward else 0.0)

    def hole_of_item(self):
        lo, hi = self.item_aabb()
        if hi[2] > -0.01:
            return None
        centre = (lo + hi) / 2
        for i, b in enumerate(BIN_POSITIONS):
            if abs(centre[0] - b[0]) < HOLE_HALF and abs(centre[1] - b[1]) < HOLE_HALF:
                return i
        return None

    def outcome(self):
        lo, hi = self.item_aabb()
        centre = (lo + hi) / 2
        hole = self.hole_of_item()
        over_hole = any(abs(centre[0] - b[0]) < HOLE_HALF and abs(centre[1] - b[1]) < HOLE_HALF for b in BIN_POSITIONS)
        if hole is not None:
            if not self.has_been_lifted:
                return "unlifted_in_hole"
            return "success" if hole == self.category else "wrong_bin"
        if lo[2] < -0.05 and not over_hole:
            return "dropped"
        if abs(centre[1]) > 0.38 or centre[0] > 0.38 or centre[0] < -0.45:
            return "dropped"
        if self.has_been_lifted and lo[2] < 0.01 and not over_hole:
            return "put_down_outside"
        return None

    def is_success(self, achieved_goal, desired_goal, info=None):
        return np.array(self.outcome() == "success", dtype=bool)

    def compute_reward(self, achieved_goal, desired_goal, info=None):
        current = self.distance_potential()
        reward = PROGRESS_SCALE * (current - self.previous_potential) - STEP_COST
        self.previous_potential = current
        if self.outcome() == "success":
            reward += SUCCESS_BONUS
        return float(reward)


class RealWasteEnv(RobotTaskEnv):
    def __init__(self, render_mode="rgb_array", randomisation=NOMINAL, items=ITEMS, n_distractors=0,
                 grasp_reward=True, regrasp=True, approach_stages=False, relative_obs=True, visual=False, dense_tray=False):
        self.curriculum_steps, self.own_steps, self.start_type = None, 0, "normal"   # set curriculum_steps for training
        self.curriculum = CurriculumState()                 # replaced by a shared one for vectorised training
        self.regrasp = regrasp
        self.stages = REVERSE_ORDER + (APPROACH_STAGES if approach_stages else ())
        self.threshold = APPROACH_THRESHOLD if approach_stages else REVERSE_THRESHOLD
        sim = PyBullet(render_mode=render_mode or "rgb_array")
        robot = PandaYaw(sim, base_position=np.array([-0.6, 0.0, 0.0]))
        task = RealWasteTask(sim, robot, randomisation, items, n_distractors, grasp_reward, relative_obs, visual, dense_tray)
        super().__init__(robot=robot, task=task, render_width=720, render_height=480,
                         render_target_position=np.array([-0.02, 0.0, 0.05]), render_distance=0.95,
                         render_yaw=60, render_pitch=-35, render_roll=0)
        del self.compute_reward                                # not a GoalEnv

    # ---------------------------------------------------------------- curriculum
    @property
    def frontier_log(self):
        return self.curriculum.log

    def _reverse_start(self):
        c = self.curriculum
        if c.frontier >= len(self.stages):
            return self.stages[int(self.np_random.integers(len(self.stages)))]
        passed = self.stages[:c.frontier]
        if passed and self.np_random.uniform() < REVERSE_REPLAY:
            return passed[int(self.np_random.integers(len(passed)))]
        return self.stages[c.frontier]

    def _record_frontier(self, success):
        c = self.curriculum
        if c.frontier >= len(self.stages) or self.start_type != self.stages[c.frontier]:
            return
        c.history = (c.history + [bool(success)])[-REVERSE_WINDOW:]
        if len(c.history) == REVERSE_WINDOW and np.mean(c.history) >= self.threshold:
            c.log.append({"passed": self.stages[c.frontier], "own step": int(c.steps)})   # all environments' steps
            c.frontier, c.history = c.frontier + 1, []

    def reset(self, seed=None, options=None):
        gym.Env.reset(self, seed=seed)
        self.task.np_random = self.np_random
        with self.sim.no_rendering():
            self.robot.reset()
            self.task.reset()
            self.start_type = (options or {}).get("start_type", "normal")
            if self.curriculum_steps:
                p_easy = CURRICULUM_P0 * max(0.0, 1.0 - self.own_steps / self.curriculum_steps)
                self.start_type = self._reverse_start() if self.np_random.uniform() < p_easy else "normal"
            self.task.apply_start(self.start_type)
        obs = self._get_obs()
        self._new_attempt()
        return obs, self._info(None)

    def _new_attempt(self):
        self.elapsed_steps, self.failed_grasp_attempts, self.previous_closed_on_nothing = 0, 0, False
        self.max_lift, self.regrasps = 0.0, 0
        self.ever_lifted = bool(self.task.has_been_lifted)

    def retarget(self, body):
        """Clearing a tray: return the arm to its start pose and make another item the target."""
        with self.sim.no_rendering():
            self.robot.reset()
        self.task.set_target(body)
        self.start_type = "normal"
        self._new_attempt()
        return self._get_obs()

    def _info(self, outcome):
        p = self.task.properties.get(self.task.target_body, {})
        return {"is_success": outcome == "success", "outcome": outcome, "item": self.task.item_name,
                "start_type": self.start_type, "category": CATEGORY_NAMES[self.task.category],
                "has_been_lifted": self.ever_lifted, "regrasps": self.regrasps,
                "dropped": outcome in ("dropped", "put_down_outside", "wrong_bin", "unlifted_in_hole"),
                "wrong_bin": outcome == "wrong_bin", "unlifted_in_hole": outcome == "unlifted_in_hole",
                "failed_grasp_attempts": self.failed_grasp_attempts, "max_lift": self.max_lift,
                "scale": p.get("scale"), "mass_scale": p.get("mass_scale"), "friction_scale": p.get("friction_scale")}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        self.elapsed_steps += 1
        self.own_steps += 1
        self.curriculum.steps += 1
        self.task.gripper_open_command = bool(action[-1] > 0.5)
        obs, reward, terminated, truncated, _ = super().step(action)
        task = self.task
        lo, hi = task.item_aabb()
        self.max_lift = max(self.max_lift, float(task.get_achieved_goal()[2] - task.rest_height))
        gripper = np.asarray(self.robot.get_ee_position())
        closed_on_nothing = aabb_distance(gripper, lo, hi) < 0.02 and self.robot.get_fingers_width() < 0.01
        if closed_on_nothing and not self.previous_closed_on_nothing:
            self.failed_grasp_attempts += 1
        self.previous_closed_on_nothing = closed_on_nothing
        self.ever_lifted = self.ever_lifted or task.has_been_lifted
        outcome = task.outcome()
        if outcome == "put_down_outside" and self.regrasp:
            # put back on the table: not a failure; the item has to be lifted again, from where it now rests
            task.has_been_lifted, task.rest_height = False, float(task.get_achieved_goal()[2])
            potential = task.distance_potential()
            reward += PROGRESS_SCALE * (potential - task.previous_potential)
            task.previous_potential = potential
            self.regrasps += 1
            obs, outcome = self._get_obs(), None
        if outcome in ("wrong_bin", "unlifted_in_hole", "dropped", "put_down_outside"):
            reward -= FAIL_PENALTY
            terminated = True
        terminated = terminated or outcome == "success"
        truncated = self.elapsed_steps >= MAX_EPISODE_STEPS and not terminated
        if self.curriculum_steps and (terminated or truncated):
            self._record_frontier(outcome == "success")
        return obs, float(reward), bool(terminated), bool(truncated), self._info(outcome)


class ObservationOnly(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = env.observation_space["observation"]

    def observation(self, observation):
        return observation["observation"].astype(np.float32)


def make_env(randomisation=NOMINAL, n_distractors=0, grasp_reward=True, regrasp=True, render_mode="rgb_array",
             curriculum_steps=None, approach_stages=False, relative_obs=True, visual=False, dense_tray=False):
    env = ObservationOnly(RealWasteEnv(render_mode=render_mode, randomisation=randomisation, n_distractors=n_distractors,
                                       grasp_reward=grasp_reward, regrasp=regrasp, approach_stages=approach_stages,
                                       relative_obs=relative_obs, visual=visual, dense_tray=dense_tray))
    env.unwrapped.curriculum_steps = curriculum_steps
    return env


class YawScripted:
    """Reference controller: align the fingers with the item's narrow side, grasp near the top, carry over the
    correct hole and let go. Exposes `phase` (0-1 approach, 2 lower, 3 close, 4 lift, 5 carry, 6 let go)."""

    def reset(self):
        self.phase, self.k = 0, 0

    def act(self, env):
        core = env.unwrapped
        task, robot, pc = core.task, core.robot, core.sim.physics_client
        lo, hi = task.item_aabb()
        centre = (lo + hi) / 2
        ee = np.asarray(robot.get_ee_position())
        hole = task.goal
        if self.phase == 0:
            self.grasp_z = max(hi[2] - 0.03, centre[2]) if hi[2] - lo[2] > 0.08 else centre[2]
            self.lift_z = self.grasp_z + (0.08 - lo[2]) + 0.04
            size = task.local_size
            yaw_item = pc.getEulerFromQuaternion(pc.getBasePositionAndOrientation(task._uid())[1])[2]
            narrow = yaw_item + (0.0 if size[0] < size[1] else np.pi / 2)
            self.yaw = float(narrow % np.pi - np.pi / 2)
            self.phase = 1
        dyaw = np.clip((self.yaw - robot.target_yaw) / MAX_YAW_STEP, -1, 1)
        move = lambda t, grip, turn=0.0: np.array([*np.clip((np.asarray(t) - ee) / 0.05, -1, 1), turn, grip], dtype=np.float32)
        if self.phase == 1:
            t = [centre[0], centre[1], hi[2] + 0.06]
            a = move(t, 1.0, dyaw)
            if np.linalg.norm(ee - t) < 0.02 and abs(self.yaw - robot.target_yaw) < 0.02:
                self.phase = 2
        elif self.phase == 2:
            t = [centre[0], centre[1], self.grasp_z]
            a = move(t, 1.0, dyaw)
            self.k += 1
            if np.linalg.norm(ee - t) < 0.015 or self.k > 40:
                self.phase, self.k = 3, 0
        elif self.phase == 3:
            a = np.array([0, 0, 0, 0, -1.0], dtype=np.float32)
            self.k += 1
            if self.k >= 8:
                self.phase = 4
        elif self.phase == 4:
            t = [ee[0], ee[1], self.lift_z]
            a = move(t, -1.0)
            if ee[2] > self.lift_z - 0.015:
                self.phase = 5
        elif self.phase == 5:
            t = [hole[0], hole[1], self.lift_z]
            a = move(t, -1.0)
            if np.linalg.norm(ee[:2] - hole[:2]) < 0.015:
                self.phase = 6
        else:
            a = np.array([0, 0, 0, 0, 1.0], dtype=np.float32)
        return a


# ---------------------------------------------------------------- evaluation
def run_episodes(choose_action, seeds, randomisation=NOMINAL, on_start=None, force_item=None, options=None, **env_kwargs):
    """One deterministic episode per seed; returns one dict per episode with stage and movement metrics."""
    env = make_env(randomisation=randomisation, **env_kwargs)
    core = env.unwrapped
    core.task.force_item = force_item
    rows = []
    for seed in seeds:
        obs, info = env.reset(seed=int(seed), options=options)
        env.action_space.seed(int(seed))
        if on_start:
            on_start()
        done, total, steps, path, switches, closest, previous_grip = False, 0.0, 0, 0.0, 0, np.inf, None
        while not done:
            before = np.asarray(core.robot.get_ee_position())
            action = choose_action(obs, env)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total += reward
            steps += 1
            path += float(np.linalg.norm(np.asarray(core.robot.get_ee_position()) - before))
            grip = bool(np.asarray(action)[-1] > 0)
            switches += int(previous_grip is not None and grip != previous_grip)
            previous_grip = grip
            lo, hi = core.task.item_aabb()
            closest = min(closest, aabb_distance(np.asarray(core.robot.get_ee_position()), lo, hi))
        rows.append({"seed": int(seed), "item": info["item"], "category": info["category"], "success": info["is_success"],
                     "outcome": info["outcome"] or "timeout", "reward": total, "length": steps,
                     "reached": closest < 0.02, "lifted": info["has_been_lifted"], "failure": info["dropped"],
                     "failed_grasp": info["failed_grasp_attempts"] > 0, "regrasps": info["regrasps"],
                     "max_lift": info["max_lift"], "path": path, "gripper_switches": switches,
                     "scale": info["scale"], "mass_scale": info["mass_scale"], "friction_scale": info["friction_scale"]})
    env.close()
    return rows


def outermost_first(task, remaining):
    # Part B rule (scripted, not learned): the item farthest from the tray centre goes first.
    distance = lambda body: np.linalg.norm(task.get_achieved_goal_of(body)[:2] - TRAY_CENTRE)
    return max(remaining, key=distance)


RealWasteTask.get_achieved_goal_of = lambda self, body: np.array(
    self.sim.physics_client.getBasePositionAndOrientation(self._uid(body))[0], dtype=np.float32)


def clear_tray(choose_action, n_items, seed, randomisation=NOMINAL, on_item_start=None, **env_kwargs):
    """Applies a single-item policy to every item on a tray of n_items (one attempt each), outermost item first."""
    env = make_env(randomisation=randomisation, n_distractors=n_items - 1, **env_kwargs)
    core = env.unwrapped
    env.reset(seed=int(seed))
    env.action_space.seed(int(seed))
    remaining, results = list(core.task.active), []
    while remaining:
        remaining = [b for b in remaining if core.task.on_table(b)]      # items knocked off or in a hole are gone
        if not remaining:
            break
        target = outermost_first(core.task, remaining)
        obs = env.observation(core.retarget(target))
        if on_item_start:
            on_item_start()
        done = False
        while not done:
            obs, reward, terminated, truncated, info = env.step(choose_action(obs, env))
            done = terminated or truncated
        results.append({"item": info["item"], "category": info["category"], "outcome": info["outcome"] or "timeout"})
        remaining.remove(target)
    env.close()
    sorted_ok = sum(r["outcome"] == "success" for r in results)
    return {"seed": int(seed), "items": len(core.task.active), "attempted": len(results), "sorted": sorted_ok,
            "whole_tray": sorted_ok == len(core.task.active), "per_item": results, "gaps m": list(core.task.tray_gaps)}
