"""
mist_env.py  –  Canonical Gymnasium environment for Project MIST.

This is the SINGLE source of truth for observation construction and
action mapping.  Both train_ppo.py and mist_controller.py (PPO eval
branch) must use this class so that training and inference see
identical input distributions.

Observation vector (shape 10, dtype float32, range [-1, 1]):
    [0..7]  8 proximity sensors, normalised to [0, 1] via /4096
    [8]     left+right mean velocity, normalised by MAX_SPEED (6.28)
    [9]     left−right diff velocity, normalised by MAX_SPEED (6.28)

Action vector (shape 2, range [-1, 1]):
    [0]  base forward speed  → wheel_avg  = action[0] * 4.0  rad/s
    [1]  angular correction  → wheel_diff = action[1] * 2.0  rad/s
    left  wheel = avg + diff
    right wheel = avg − diff
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np

# E-puck physical constants
MAX_WHEEL_SPEED = 6.28          # rad/s  (hardware limit)
PROX_MAX        = 4096.0        # maximum proximity sensor ADC reading
TRACK_WIDTH     = 0.053         # metres between wheel centres (used elsewhere)

# Lane layout  (x_start, y_centre, x_goal)
LANE_STARTS = [
    [-8.8541,  3.13013, 0.0],
    [-8.8541,  0.370129, 0.0],
    [-8.8541, -2.36987, 0.0],
    [-8.8541, -5.12987, 0.0],
]
GOAL_X = 10.6659


class MistNavEnv(gym.Env):
    """
    Gymnasium wrapper around the Webots Supervisor API for the MIST
    four-lane navigation benchmark.

    Parameters
    ----------
    robot_supervisor : webots Supervisor instance
    wheels           : list of two Motor devices [left, right]
    proximity_sensors: list of 8 DistanceSensor devices [ps0..ps7]
    noise_level      : std-dev of Gaussian noise injected on prox readings
                       (0.0 = clean, 0.02 = heavy fog emulation in ADC units
                        relative to the normalised [0,1] scale)
    """

    metadata = {"render_modes": []}

    def __init__(self, robot_supervisor, wheels, proximity_sensors,
                 noise_level: float = 0.0):
        super().__init__()
        self.robot            = robot_supervisor
        self.wheels           = wheels
        self.proximity_sensors = proximity_sensors
        self.timestep         = int(self.robot.getBasicTimeStep())
        self.robot_node       = self.robot.getSelf()
        self.noise_level      = noise_level

        self.step_count = 0
        self.MAX_STEPS  = 3000
        self.prev_dist  = 0.0
        self.current_lane = 0

        # ── Action space ──────────────────────────────────────────────
        # [forward_cmd, angular_cmd], both in [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        # ── Observation space ─────────────────────────────────────────
        # 8 normalised prox readings in [0,1] + 2 velocity signals in [-1,1]
        # We declare the full range as [-1,1] so SB3 does not clip/warn on
        # the velocity channels.
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(10,), dtype=np.float32
        )

    # ──────────────────────────────────────────────────────────────────
    # Public helpers
    # ──────────────────────────────────────────────────────────────────

    def set_noise(self, noise_level: float):
        """Change the simulated fog / backscatter noise level at runtime."""
        self.noise_level = noise_level

    # ──────────────────────────────────────────────────────────────────
    # Core Gymnasium interface
    # ──────────────────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        """
        Build a normalised observation vector.

        Proximity sensors:
            Raw ADC value in [0, 4096]. Normalised to [0, 1] by dividing by
            PROX_MAX.  Gaussian noise (std = self.noise_level) is added in
            normalised units and the result is clipped to [0, 1].

        Velocity channels:
            Both normalised to [-1, 1] by dividing by MAX_WHEEL_SPEED.
        """
        noise = np.random.normal(0.0, self.noise_level, 8) if self.noise_level > 0 else np.zeros(8)

        prox = np.array(
            [np.clip(ps.getValue() / PROX_MAX + n, 0.0, 1.0)
             for ps, n in zip(self.proximity_sensors, noise)],
            dtype=np.float32
        )

        v_l = self.wheels[0].getVelocity()
        v_r = self.wheels[1].getVelocity()
        v_mean  = np.clip((v_l + v_r) / (2.0 * MAX_WHEEL_SPEED), -1.0, 1.0)
        v_diff  = np.clip((v_l - v_r) / (2.0 * MAX_WHEEL_SPEED), -1.0, 1.0)

        return np.concatenate([prox, [v_mean, v_diff]]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0

        start_pos = LANE_STARTS[self.current_lane]
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

        # Map actions to wheel velocities
        avg  = float(action[0]) * 4.0   # max 4 rad/s forward
        diff = float(action[1]) * 2.0   # max 2 rad/s turn

        v_l = np.clip(avg + diff, -MAX_WHEEL_SPEED, MAX_WHEEL_SPEED)
        v_r = np.clip(avg - diff, -MAX_WHEEL_SPEED, MAX_WHEEL_SPEED)
        self.wheels[0].setVelocity(v_l)
        self.wheels[1].setVelocity(v_r)

        self.robot.step(self.timestep)

        obs = self._get_obs()
        pos = self.robot_node.getPosition()

        # ── Reward ────────────────────────────────────────────────────
        curr_dist = abs(GOAL_X - pos[0])
        progress  = self.prev_dist - curr_dist
        self.prev_dist = curr_dist

        # Collision penalty: prox[0..7] normalised; value > 0.15 is close contact
        front_sensors = [obs[0], obs[1], obs[6], obs[7]]   # ps0,ps1,ps6,ps7
        collision = any(v > 0.15 for v in front_sensors)

        reward  =  progress * 20.0          # forward progress reward
        reward -= (5.0 if collision else 0.0)  # collision penalty
        reward -= abs(action[1]) * 0.1      # steering smoothness penalty

        # ── Termination ───────────────────────────────────────────────
        terminated = pos[0] >= GOAL_X
        truncated  = self.step_count >= self.MAX_STEPS

        if terminated:
            reward += 100.0
            self.current_lane = (self.current_lane + 1) % 4

        return obs, float(reward), terminated, truncated, {}