"""
train_ppo.py  –  Trains the PPO agent for Project MIST inside Webots.

Run via the Webots extern-controller mechanism:
    MIST_NAV_MODE=PURE_PPO  python train_ppo.py

The script bootstraps the Supervisor, constructs a MistNavEnv wrapping the
live Webots session, then hands control to Stable-Baselines3's PPO trainer.
After training the model is saved to  <project_root>/saved_models/ppo_mist_optimal_model
"""

import os
import sys
from pathlib import Path

from controller import Supervisor
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

# ── Path resolution ───────────────────────────────────────────────────────────
# Folder layout expected:
#   <project_root>/
#       envs/worlds/worlds/         ← this file lives here (extern controller)
#       src/utils/mist_env.py
#       saved_models/

SRC_DIR   = Path(__file__).resolve().parent            # .../envs/worlds/worlds
# Walk up to project root: worlds/ → worlds/ → envs/ → <root>
MIST_ROOT = SRC_DIR.parent.parent.parent

# Make utils importable
UTILS_DIR = MIST_ROOT / "src" / "utils"
sys.path.insert(0, str(UTILS_DIR))

from utils.mist_env import MistNavEnv


# ── Optional noise-curriculum callback ───────────────────────────────────────
class NoiseCurriculumCallback(BaseCallback):
    """
    Gradually increases sensor noise during training to make the agent
    robust to fog / backscatter (the core MIST hypothesis).

    Noise schedule (in normalised prox units):
        0 – 30 k steps  : 0.00  (clean environment, learn basic navigation)
       30 – 70 k steps  : 0.01  (light fog)
       70 k+ steps      : 0.02  (heavy fog / backscatter)
    """

    def __init__(self, env: MistNavEnv, verbose: int = 1):
        super().__init__(verbose)
        self.env = env

    def _on_step(self) -> bool:
        n = self.num_timesteps
        if n < 30_000:
            self.env.set_noise(0.00)
        elif n < 70_000:
            self.env.set_noise(0.01)
        else:
            self.env.set_noise(0.02)
        return True


def main():
    # ── Webots hardware init ──────────────────────────────────────────────────
    robot    = Supervisor()
    timestep = int(robot.getBasicTimeStep())

    wheels = [robot.getDevice("left wheel motor"),
              robot.getDevice("right wheel motor")]
    for w in wheels:
        w.setPosition(float('inf'))
        w.setVelocity(0.0)

    proximity_sensors = []
    for i in range(8):
        ps = robot.getDevice(f"ps{i}")
        ps.enable(timestep)
        proximity_sensors.append(ps)

    # ── Environment ───────────────────────────────────────────────────────────
    env = MistNavEnv(robot, wheels, proximity_sensors, noise_level=0.0)

    # ── Model ─────────────────────────────────────────────────────────────────
    tensorboard_dir = str(MIST_ROOT / "ppo_mist_tensorboard")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,          # small entropy bonus for exploration
        tensorboard_log=tensorboard_dir,
    )

    curriculum_cb = NoiseCurriculumCallback(env, verbose=1)

    # ── Training ──────────────────────────────────────────────────────────────
    TOTAL_STEPS = 150_000   # ~150 k steps give a reasonable policy; increase for publication
    print(f"🚀 Training starting – {TOTAL_STEPS:,} timesteps with noise curriculum …")
    model.learn(total_timesteps=TOTAL_STEPS, callback=curriculum_cb)

    # ── Save ──────────────────────────────────────────────────────────────────
    save_dir = MIST_ROOT / "saved_models"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "ppo_mist_optimal_model"
    model.save(str(save_path))
    print(f"✅ Training complete.  Model saved to: {save_path}.zip")

    robot.simulationQuit(0)


if __name__ == "__main__":
    main()