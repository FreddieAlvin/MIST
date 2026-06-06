"""
train_ppo.py – Trains the PPO agent for Project MIST.
"""

import os
import sys
from pathlib import Path

# ── 0. Set Environment Variables BEFORE importing controller ──
# Manually set WEBOTS_HOME
os.environ['WEBOTS_HOME'] = '/Applications/Webots.app'

# Add the library to sys.path
webots_path = os.path.join(os.environ['WEBOTS_HOME'], 'Contents', 'lib', 'controller', 'python')
if webots_path not in sys.path:
    sys.path.append(webots_path)

# Now it is safe to import Webots & Stable-Baselines3 elements
from controller import Supervisor
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

# ── 1. Path Resolution ──
CURRENT_DIR = Path(__file__).resolve().parent        # Points to MIST/src/agents/
SRC_DIR     = CURRENT_DIR.parent                       # Points to MIST/src/
MIST_ROOT   = SRC_DIR.parent                           # Points to MIST/ root folder
LOG_DIR     = MIST_ROOT / "results" / "logs"
TB_DIR      = MIST_ROOT / "results" / "tensorboard"
MODEL_DIR   = MIST_ROOT / "saved_models"               # Aligned with mist_controller.py lookup path

for folder in [LOG_DIR, TB_DIR, MODEL_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# Safe search path additions for structural flexibility
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(SRC_DIR / "utils"))

from mist_env import MistNavEnv

# ── 2. Noise-Curriculum Callback ──
class NoiseCurriculumCallback(BaseCallback):
    def __init__(self, env: MistNavEnv, verbose: int = 1):
        super().__init__(verbose)
        self.env = env

    def _on_step(self) -> bool:
        n = self.num_timesteps
        if n < 30_000:   self.env.set_noise(0.00)
        elif n < 70_000: self.env.set_noise(0.01)
        else:            self.env.set_noise(0.02)
        return True

def main():
    # ── Webots Hardware Init ──
    robot    = Supervisor()
    timestep = int(robot.getBasicTimeStep())
    wheels   = [robot.getDevice("left wheel motor"), robot.getDevice("right wheel motor")]
    for w in wheels:
        w.setPosition(float('inf'))
        w.setVelocity(0.0)

    proximity_sensors = [robot.getDevice(f"ps{i}") for i in range(8)]
    for ps in proximity_sensors:
        ps.enable(timestep)

    # ── Environment ──
    env = MistNavEnv(robot, wheels, proximity_sensors, noise_level=0.0)

    # ── PPO Model Setup ──
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
        ent_coef=0.01,
        tensorboard_log=str(TB_DIR),
    )

    curriculum_cb = NoiseCurriculumCallback(env, verbose=1)

    # ── Training ──
    TOTAL_STEPS = 150_000
    print(f"🚀 Training starting: {TOTAL_STEPS:,} steps...")
    model.learn(total_timesteps=TOTAL_STEPS, callback=curriculum_cb)

    # ── Save Artifacts ──
    save_path = MODEL_DIR / "ppo_mist_optimal_model"
    model.save(str(save_path))
    print(f"✅ Training complete. Model saved to: {save_path}.zip")

    robot.simulationQuit(0)

if __name__ == "__main__":
    main()