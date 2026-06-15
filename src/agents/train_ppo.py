"""
train_ppo.py – Trains the PPO agent for Project MIST.

Feature-fusion architecture:
    • A NatureCNN encodes the camera "image" key (uint8 → /255 inside the CNN).
    • SB3's CombinedExtractor flattens the "lidar" and "vector" keys and
      concatenates them with the CNN features.
    • The fused vector then feeds the shared Actor-Critic MLP (net_arch).
    (MultiInputPolicy wires all of this automatically for a Dict obs space.)

Noise curriculum (applied to the lidar/proximity channel only):
    0 – 30k steps  : noise = 0.00  (clean sensors)
    30k – 70k steps: noise = 0.01  (light fog)
    70k+   steps   : noise = 0.02  (heavy fog)
"""

import os
import sys
from pathlib import Path

# ── 0. Webots path (cross-platform) ──────────────────────────────────────────
if sys.platform == "darwin":
    os.environ['WEBOTS_HOME'] = '/Applications/Webots.app'
    _wlib = os.path.join(os.environ['WEBOTS_HOME'],
                         'Contents', 'lib', 'controller', 'python')
elif sys.platform == "win32":
    os.environ['WEBOTS_HOME'] = 'C:\\Program Files\\Webots'
    _wlib = os.path.join(os.environ['WEBOTS_HOME'],
                         'lib', 'controller', 'python')
else:   # Linux
    os.environ['WEBOTS_HOME'] = '/usr/local/webots'
    _wlib = os.path.join(os.environ['WEBOTS_HOME'],
                         'lib', 'controller', 'python')

if _wlib not in sys.path:
    sys.path.append(_wlib)

# ── 1. Path resolution ────────────────────────────────────────────────────────
CURRENT_DIR = Path(__file__).resolve().parent    # MIST/src/agents/
SRC_DIR     = CURRENT_DIR.parent                  # MIST/src/
MIST_ROOT   = SRC_DIR.parent                      # MIST/
LOG_DIR     = MIST_ROOT / "results" / "logs"
TB_DIR      = MIST_ROOT / "results" / "tensorboard"
MODEL_DIR   = MIST_ROOT / "saved_models"

for folder in [LOG_DIR, TB_DIR, MODEL_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(SRC_DIR / "utils"))

# ── 2. Imports ────────────────────────────────────────────────────────────────
from controller import Supervisor
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from mist_env import MistNavEnv

# ── 3. Noise-curriculum callback ──────────────────────────────────────────────
class NoiseCurriculumCallback(BaseCallback):
    """
    Sensor-noise curriculum (fog robustness): 0.00 → 0.01 → 0.02, applied to the
    *unwrapped* MistNavEnv. (The reverse start-distance curriculum was removed —
    it destabilised transfer to the true start.)
    """
    def __init__(self, env: MistNavEnv, total_steps: int = 0, verbose: int = 1):
        super().__init__(verbose)
        self.nav_env = env
        self._last_band = None

    def _on_step(self) -> bool:
        n = self.num_timesteps
        if n < 30_000:
            band, noise = "clean", 0.00
        elif n < 70_000:
            band, noise = "light_fog", 0.01
        else:
            band, noise = "heavy_fog", 0.02
        self.nav_env.set_noise(noise)
        if self.verbose and band != self._last_band:
            print(f"🌫️  Noise curriculum → {band} (σ={noise}) @ {n:,} steps")
            self._last_band = band
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

    proximity_sensors = [robot.getDevice(f"ps{i}") for i in range(8)]
    for ps in proximity_sensors:
        ps.enable(timestep)

    # Lidar (360-ray, 3 m) – the policy's main obstacle sensor.
    lidar = robot.getDevice("lidar")
    if lidar:
        lidar.enable(timestep)
        print(f"✅ Lidar enabled ({lidar.getHorizontalResolution()} rays, "
              f"{lidar.getMaxRange():.1f} m range)")
    else:
        print("⚠️  No Lidar found – 'lidar' observation will be all-clear.")

    # Camera (optional – the env gracefully falls back to a zero image)
    camera = robot.getDevice("camera")
    if camera:
        camera.enable(timestep)
        print(f"✅ Camera enabled ({camera.getWidth()}×{camera.getHeight()} px)")
    else:
        print("⚠️  No camera found – image observations will be all-zero tensors.")

    # ── Environment ───────────────────────────────────────────────────────────
    env = MistNavEnv(
        robot_supervisor=robot,
        wheels=wheels,
        proximity_sensors=proximity_sensors,
        camera=camera,
        lidar=lidar,
        noise_level=0.0,
    )

    # ── PPO with MultiInputPolicy (NatureCNN + MLP fusion) ────────────────────
    # net_arch defines the shared Actor-Critic MLP AFTER the CombinedExtractor
    # has fused [CNN(image) ⊕ lidar ⊕ vector].
    # FIX: modern SB3 (≥2.0) expects a dict here, not a [dict] list.
    policy_kwargs = dict(
        net_arch=dict(pi=[256, 128], vf=[256, 128]),
    )

    model = PPO(
        "MultiInputPolicy",       # NatureCNN for image + flatten for vectors
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.02,            # ↑ from 0.01: keep exploration, resist std collapse
        target_kl=0.03,           # early-stop runaway updates (saw approx_kl=0.15)
        policy_kwargs=policy_kwargs,
        tensorboard_log=str(TB_DIR),
    )

    # ── Training ──────────────────────────────────────────────────────────────
    # Episodes are long (~5k steps), so give PPO enough transitions to see many
    # completed lanes across all four lane positions.
    TOTAL_STEPS = 600_000
    curriculum_cb = NoiseCurriculumCallback(env, total_steps=TOTAL_STEPS, verbose=1)

    print(f"🚀 Training starting: {TOTAL_STEPS:,} steps …")
    print("   Policy: MultiInputPolicy (NatureCNN camera + Lidar + kinematics)")
    model.learn(total_timesteps=TOTAL_STEPS, callback=curriculum_cb)

    # ── Save ──────────────────────────────────────────────────────────────────
    save_path = MODEL_DIR / "ppo_mist_optimal_model"
    model.save(str(save_path))
    print(f"✅ Training complete. Model saved to: {save_path}.zip")

    robot.simulationQuit(0)


if __name__ == "__main__":
    main()