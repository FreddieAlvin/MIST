"""
mist_env.py  –  Canonical Gymnasium environment for Project MIST.

This is the SINGLE source of truth for observation construction and
action mapping.  Both train_ppo.py and mist_controller.py (PPO eval
branch) MUST use this class / these constants so that training and
inference see identical input distributions.

Observation (Dict, for MultiInputPolicy → CombinedExtractor):
    "lidar"  : shape (8,),  dtype float32, range [0, 1]
               8 proximity sensors normalised via /PROX_MAX
    "image"  : shape (H, W, 3), dtype UINT8, range [0, 255]   ← visual stream
               Camera frame resampled to IMAGE_H × IMAGE_W.
               *** Must be uint8 [0,255] so Stable-Baselines3 routes it
                   through a NatureCNN. A float [0,1] image is NOT detected
                   as an image and would be flattened into the MLP. ***
               Falls back to a zero tensor when no camera is present.
    "vector" : shape (6,),  dtype float32, range [-1, 1]
               [0]  mean wheel velocity,      normalised by MAX_WHEEL_SPEED
               [1]  diff wheel velocity,      normalised by MAX_WHEEL_SPEED
               [2]  heading error to goal,    normalised by π (→ [-1,1])
               [3]  relative X to goal,       normalised by 20.0 m
               [4]  relative Y to goal,       normalised by 20.0 m
               [5]  distance to goal (scalar) normalised by GOAL_DIST_NORM

Action vector (shape 2, range [-1, 1]):
    [0]  base forward speed  → wheel_avg  = action[0] * FORWARD_SCALE  rad/s
    [1]  angular correction  → wheel_diff = action[1] * TURN_SCALE     rad/s
    left  wheel = avg + diff
    right wheel = avg − diff

Step info dict keys:
    "lane_completed" : bool  – True when the robot reached the goal X
    "lane_index"     : int   – which of the 4 lanes was active this episode
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np

# ── E-puck physical constants ─────────────────────────────────────────────────
MAX_WHEEL_SPEED = 6.28       # rad/s  (hardware limit, matches .wbt maxVelocity)
WHEEL_RADIUS    = 0.02       # m      (matches .wbt wheel Cylinder radius)
PROX_MAX        = 4096.0     # maximum proximity sensor ADC reading
TRACK_WIDTH     = 0.053      # metres between wheel centres

# ── Lidar (planning sensor) ───────────────────────────────────────────────────
# The .wbt defines a 360-ray, 360° FOV, 3 m-range Lidar that was previously
# unused. It is down-sampled into LIDAR_SECTORS arcs (nearest hit per arc) and
# becomes the "lidar" observation key, giving the policy ~30× the lookahead of
# the 10 cm proximity sensors. Proximity sensors are kept for collision flagging.
LIDAR_SECTORS   = 16
LIDAR_MAX_RANGE = 3.0        # metres (matches .wbt maxRange)

# ── Action scaling (shared by train + controller; do NOT diverge!) ────────────
# FIX: forward scale was 4.0 rad/s → max 0.08 m/s → the 19.5 m lane needed
# ~244 s (7600 steps) to cross, but MAX_STEPS was 5000. The goal was therefore
# physically unreachable, so the agent never terminated, never collected the
# completion bonus, and just farmed the per-step forward reward.
FORWARD_SCALE = MAX_WHEEL_SPEED   # action[0] * FORWARD_SCALE → avg wheel (rad/s)
TURN_SCALE    = 3.0               # action[1] * TURN_SCALE    → diff wheel (rad/s)

# ── Camera resolution (downsampled for policy network) ────────────────────────
IMAGE_H = 64   # pixels – matches NatureCNN expectation
IMAGE_W = 64

# ── Camera toggle ─────────────────────────────────────────────────────────────
# Camera ON. The camera-on run was the high-water mark (cleared lanes 0 AND 1,
# weaving lane 1's 10 obstacles). Now fused WITH the 16-sector Lidar (3 m
# lookahead), the policy gets both visual context and ranged obstacle geometry —
# richer than either alone, and faithful to the original camera+lidar proposal.
USE_CAMERA = True

# ── Lane layout (must match mist_controller.py) ───────────────────────────────
LANE_STARTS = [
    [-8.8541,  3.13013,  -0.000217358],   # Lane 0
    [-8.8541,  0.370129, -0.000217358],   # Lane 1
    [-8.8541, -2.36987,  -0.000217358],   # Lane 2
    [-8.8541, -5.12987,  -0.000217358],   # Lane 3
]
GOAL_X   = 10.6659
GOAL_DIST_NORM = 25.0   # normalisation denominator for distance-to-goal

# ── Episode budget ────────────────────────────────────────────────────────────
# At full speed (0.126 m/s) a clean lane takes ~155 s ≈ 4860 steps. 8000 steps
# (~256 s) leaves comfortable margin for turning / obstacle avoidance.
MAX_EPISODE_STEPS = 8000


class MistNavEnv(gym.Env):
    """
    Gymnasium wrapper around the Webots Supervisor API for the MIST
    four-lane navigation benchmark.

    Dict observation space so SB3's MultiInputPolicy / CombinedExtractor can:
        • route the uint8 "image" key through a NatureCNN
        • flatten "lidar" + "vector" and concatenate them with the CNN
          features before the shared Actor-Critic MLP (net_arch).
    """

    metadata = {"render_modes": []}

    def __init__(self, robot_supervisor, wheels, proximity_sensors,
                 camera=None, lidar=None, noise_level: float = 0.0):
        super().__init__()
        self.robot             = robot_supervisor
        self.wheels            = wheels
        self.proximity_sensors = proximity_sensors
        self.camera            = camera
        self.lidar             = lidar
        self.timestep          = int(self.robot.getBasicTimeStep())
        self.robot_node        = self.robot.getSelf()
        self.noise_level       = noise_level

        self.step_count   = 0
        self.MAX_STEPS    = MAX_EPISODE_STEPS
        self.prev_dist    = 0.0
        self.current_lane = 0

        # ── Action space ──────────────────────────────────────────────────────
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        # ── Observation space (Dict for MultiInputPolicy) ─────────────────────
        obs_spaces = {
            # LIDAR_SECTORS down-sampled Lidar ranges in [0, 1] (1 = clear @ 3 m)
            "lidar": spaces.Box(
                low=0.0, high=1.0, shape=(LIDAR_SECTORS,), dtype=np.float32
            ),
            # Kinematic / goal vector in [-1, 1]
            "vector": spaces.Box(
                low=-1.0, high=1.0, shape=(6,), dtype=np.float32
            ),
        }
        if USE_CAMERA:
            # Camera frame as uint8 [0,255] → detected as an image by SB3 and
            # routed through a NatureCNN.
            obs_spaces["image"] = spaces.Box(
                low=0, high=255, shape=(IMAGE_H, IMAGE_W, 3), dtype=np.uint8
            )
        self.observation_space = spaces.Dict(obs_spaces)

    # ──────────────────────────────────────────────────────────────────────────
    # Public helpers
    # ──────────────────────────────────────────────────────────────────────────

    def set_noise(self, noise_level: float):
        """Change the simulated fog / backscatter noise level at runtime."""
        self.noise_level = noise_level

    # ──────────────────────────────────────────────────────────────────────────
    # Observation construction
    # ──────────────────────────────────────────────────────────────────────────

    def _get_prox(self) -> np.ndarray:
        """8 proximity readings normalised to [0,1] (used for collision flag)."""
        return np.array(
            [np.clip(ps.getValue() / PROX_MAX, 0.0, 1.0)
             for ps in self.proximity_sensors],
            dtype=np.float32
        )

    def _get_lidar_sectors(self) -> np.ndarray:
        """
        Down-sample the 360-ray Lidar into LIDAR_SECTORS arcs (min range per arc,
        i.e. nearest obstacle in that direction), normalised to [0,1] where
        1 = clear at LIDAR_MAX_RANGE and ~0 = obstacle right next to the robot.
        Fog noise (the curriculum) is applied here. Falls back to "all clear"
        if no Lidar device is present.
        """
        if self.lidar is None:
            return np.ones(LIDAR_SECTORS, dtype=np.float32)

        ranges = np.asarray(self.lidar.getRangeImage(), dtype=np.float32)
        if ranges.size == 0:
            return np.ones(LIDAR_SECTORS, dtype=np.float32)

        # No-hit rays come back as inf → treat as max range (fully clear).
        ranges = np.where(np.isfinite(ranges), ranges, LIDAR_MAX_RANGE)
        ranges = np.clip(ranges, 0.0, LIDAR_MAX_RANGE)

        # Nearest obstacle per sector, then normalise.
        sectors = np.array([float(np.min(g))
                            for g in np.array_split(ranges, LIDAR_SECTORS)],
                           dtype=np.float32)
        norm = sectors / LIDAR_MAX_RANGE

        if self.noise_level > 0:
            norm = np.clip(norm + np.random.normal(0.0, self.noise_level,
                                                   LIDAR_SECTORS), 0.0, 1.0)
        return norm.astype(np.float32)

    def _get_image(self) -> np.ndarray:
        """
        Grab one camera frame, resample to IMAGE_H × IMAGE_W, return uint8.
        Returns a zero tensor if no camera is connected.
        """
        if self.camera is None:
            return np.zeros((IMAGE_H, IMAGE_W, 3), dtype=np.uint8)

        raw = self.camera.getImage()   # bytes: BGRA interleaved
        w   = self.camera.getWidth()
        h   = self.camera.getHeight()

        if raw is None or len(raw) == 0:
            return np.zeros((IMAGE_H, IMAGE_W, 3), dtype=np.uint8)

        # Convert BGRA bytes → RGB uint8 numpy array
        arr  = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 4))
        rgb  = arr[:, :, 2::-1]   # drop alpha, reverse BGR→RGB

        # Nearest-neighbour resample to IMAGE_H × IMAGE_W (handles up/down scale)
        row_idx = (np.arange(IMAGE_H) * h / IMAGE_H).astype(int)
        col_idx = (np.arange(IMAGE_W) * w / IMAGE_W).astype(int)
        frame   = rgb[np.ix_(row_idx, col_idx)]   # (IMAGE_H, IMAGE_W, 3)

        # Keep uint8 [0,255] – SB3 normalises by /255 inside the CNN.
        return np.ascontiguousarray(frame, dtype=np.uint8)

    def _get_vector(self) -> np.ndarray:
        """Build the 6-feature kinematic / goal vector."""
        v_l    = self.wheels[0].getVelocity()
        v_r    = self.wheels[1].getVelocity()
        v_mean = np.clip((v_l + v_r) / (2.0 * MAX_WHEEL_SPEED), -1.0, 1.0)
        v_diff = np.clip((v_l - v_r) / MAX_WHEEL_SPEED,          -1.0, 1.0)

        # Robot pose
        pos        = self.robot_node.getPosition()
        rot_matrix = self.robot_node.getOrientation()
        current_yaw = np.arctan2(rot_matrix[3], rot_matrix[0])   # atan2(r10, r00)

        goal_x = GOAL_X
        goal_y = LANE_STARTS[self.current_lane][1]

        # Heading error to goal, normalised to [-1, 1]
        target_angle  = np.arctan2(goal_y - pos[1], goal_x - pos[0])
        heading_error = (target_angle - current_yaw + np.pi) % (2 * np.pi) - np.pi
        heading_norm  = np.clip(heading_error / np.pi, -1.0, 1.0)

        # Relative X and Y distances, normalised
        rel_x = np.clip((goal_x - pos[0]) / 20.0, -1.0, 1.0)
        rel_y = np.clip((goal_y - pos[1]) / 20.0, -1.0, 1.0)

        # Euclidean distance to goal, normalised
        dist_to_goal = np.hypot(goal_x - pos[0], goal_y - pos[1])
        dist_norm    = np.clip(dist_to_goal / GOAL_DIST_NORM, 0.0, 1.0)

        return np.array(
            [v_mean, v_diff, heading_norm, rel_x, rel_y, dist_norm],
            dtype=np.float32
        )

    def _get_obs(self) -> dict:
        lidar  = self._get_lidar_sectors()
        vector = self._get_vector()
        obs = {"lidar": lidar, "vector": vector}
        if USE_CAMERA:
            obs["image"] = self._get_image()
        return obs

    # ──────────────────────────────────────────────────────────────────────────
    # Core Gymnasium interface
    # ──────────────────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0

        # Randomise lane every episode so all four lanes are trained.
        self.current_lane = int(self.np_random.integers(0, len(LANE_STARTS)))
        lane = LANE_STARTS[self.current_lane]

        # Spawn at the true lane start. (The reverse curriculum was removed: its
        # frontier schedule gave healthy training metrics but a policy that did
        # not transfer to the true start — adding complexity hurt more than it
        # helped. Back to the simple, stable setup of the best run.)
        start_pos = [lane[0], lane[1], lane[2]]

        self.robot_node.getField("translation").setSFVec3f(start_pos)
        self.robot_node.getField("rotation").setSFRotation([0, 0, 1, 0])
        self.robot_node.resetPhysics()

        for w in self.wheels:
            w.setVelocity(0.0)

        self.robot.step(self.timestep)
        self.prev_dist = abs(GOAL_X - start_pos[0])
        return self._get_obs(), {}

    def step(self, action):
        self.step_count += 1

        # Map actions to wheel velocities (shared scaling constants)
        avg   = float(action[0]) * FORWARD_SCALE
        diff  = float(action[1]) * TURN_SCALE
        v_l   = np.clip(avg + diff, -MAX_WHEEL_SPEED, MAX_WHEEL_SPEED)
        v_r   = np.clip(avg - diff, -MAX_WHEEL_SPEED, MAX_WHEEL_SPEED)
        self.wheels[0].setVelocity(v_l)
        self.wheels[1].setVelocity(v_r)

        self.robot.step(self.timestep)

        obs = self._get_obs()
        pos = self.robot_node.getPosition()

        # ── 1. Progress reward (telescoping: Σ progress = total X displacement)─
        curr_dist = abs(GOAL_X - pos[0])
        progress  = self.prev_dist - curr_dist
        self.prev_dist = curr_dist

        # ── 2. Collision detection (front proximity sensors ps0,ps1,ps6,ps7) ──
        prox = self._get_prox()
        front_sensors = [prox[0], prox[1], prox[6], prox[7]]
        collision = any(v > 0.45 for v in front_sensors)

        # ── 3. Reward shaping ─────────────────────────────────────────────────
        # v3 FIX: the v2 shaping caused a PPO policy collapse. The `-5.0`
        # collision term fired EVERY step the robot touched a wall, so early
        # random exploration made "move" strongly negative and "freeze" the
        # least-bad action → entropy collapsed (std 1.0 → 0.13). Rebalanced so
        # forward progress is the dominant signal and collision is a moderate
        # nudge, not a per-step cliff. (PPO normalises advantages, so it's the
        # RATIO of these terms that matters, not their absolute scale.)
        reward  = progress * 50.0                  # dominant dense signal
        reward -= 0.01                             # small time penalty
        if collision:
            reward -= 0.5                          # good-run value (weave > graze)
        reward -= abs(float(action[1])) * 0.02     # gentle steering penalty
        # NOTE: an anti-freeze penalty was tried (v4 on action[0], v5 on actual
        # displacement) to stop the lane-2/3 freeze. Both HURT: v4 taught the
        # policy to spin in place; v5 added enough reward noise that PPO never
        # learned (std stuck at 0.92 after 300k). Reverted. The real cause of
        # the hard-lane freeze is insufficient sensing, addressed via the Lidar.

        # ── Termination / truncation ──────────────────────────────────────────
        lane_completed = bool(pos[0] >= GOAL_X)
        terminated     = lane_completed
        truncated      = self.step_count >= self.MAX_STEPS

        if terminated:
            reward += 200.0   # completion bonus (now actually reachable)

        info = {
            "lane_completed": lane_completed,
            "lane_index":     self.current_lane,
        }

        return obs, float(reward), terminated, truncated, info