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
    os.environ['WEBOTS_HOME'] = r'C:\Users\ines.castro\Desktop\robotica\Webots'
    _wlib = os.path.join(os.environ['WEBOTS_HOME'], 'lib', 'controller', 'python')
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
from stable_baselines3.common.monitor import Monitor
from mist_env import MistNavEnv

# ── 3. Noise-curriculum callback ──────────────────────────────────────────────
# SHOULD TRAINING INCLUDE NOISE ("mist")?
#   • Project MIST is about navigating under degraded ("foggy") sensing, so the
#     scientifically correct setup is DOMAIN RANDOMISATION: train with noise so
#     the policy learns to be robust, then *test* at several noise levels to
#     measure the robustness curve. A policy trained only on clean sensors will
#     degrade sharply when you add mist at test time.
#   • BUT noise also makes the task harder to learn. So we learn the task clean
#     first (0–30k), then ramp in light/heavy fog. If PPO is struggling to learn
#     the task at all, set TRAIN_WITH_NOISE=False to debug on clean sensors, get
#     it completing lanes, then turn noise back on for the final robust model.
TRAIN_WITH_NOISE = os.environ.get("MIST_TRAIN_NOISE", "0").lower() in ("1", "true", "yes")
# Model name → used both for the saved .zip and the Monitor log, so training a
# second (fog) model never overwrites the baseline model or its learning curve.
MODEL_NAME = os.environ.get("MIST_MODEL", "ppo_mist_optimal_model")

class NoiseCurriculumCallback(BaseCallback):
    """
    Two curricula on the *unwrapped* MistNavEnv:
      • Reverse start-distance curriculum (always on): frac_max 0.3→1.0 over the
        first half of training — learn to finish from near the goal, then from
        progressively further back.
      • Optional fog-noise curriculum (TRAIN_WITH_NOISE): 0.00→0.01→0.02.
    """
    def __init__(self, env: MistNavEnv, total_steps: int = 1, verbose: int = 1):
        super().__init__(verbose)
        self.nav_env = env
        self.total_steps = max(1, total_steps)
        self._last_band = None

    def _on_step(self) -> bool:
        n = self.num_timesteps

        # Reverse curriculum (reach full lane at 50% of training)
        ramp_end = max(1, int(0.5 * self.total_steps))
        self.nav_env.set_start_curriculum(min(1.0, 0.3 + 0.7 * n / ramp_end))

        # Optional fog noise
        if not TRAIN_WITH_NOISE:
            self.nav_env.set_noise(0.0)
            return True
        if n < 20_000:
            band, noise = "clean", 0.00
        elif n < 50_000:
            band, noise = "light_fog", 0.02  # Sobe para o nível intermédio
        else:
            band, noise = "heavy_fog", 0.05
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
    nav_env = MistNavEnv(
        robot_supervisor=robot,
        wheels=wheels,
        proximity_sensors=proximity_sensors,
        camera=camera,
        lidar=lidar,
        noise_level=0.0,
    )

    # Monitor writes per-episode reward/length to results/logs/ppo_monitor.monitor.csv
    # → plot_results.py turns this into the PPO learning curve. The callback keeps
    # a handle to the RAW env (nav_env) so it can drive the curricula directly.
    env = Monitor(nav_env, filename=str(LOG_DIR / f"{MODEL_NAME}_monitor"))

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
    TOTAL_STEPS = 400_000
    curriculum_cb = NoiseCurriculumCallback(nav_env, total_steps=TOTAL_STEPS, verbose=1)

    print(f"🚀 Training starting: {TOTAL_STEPS:,} steps …")
    print("   Policy: MultiInputPolicy (Lidar + kinematics; camera off) + clearance")
    print(f"   Train lanes: {__import__('mist_env').TRAIN_LANES} "
          "(all static in training; lane 3 dynamic only at test)")
    print(f"   Reverse curriculum: ON | training noise: {TRAIN_WITH_NOISE}")
    model.learn(total_timesteps=TOTAL_STEPS, callback=curriculum_cb)

    # ── Save ──────────────────────────────────────────────────────────────────
    # MIST_MODEL lets you keep a clean baseline model AND a fog-trained one:
    #   baseline :  python train_ppo.py
    #   robust   :  $env:MIST_TRAIN_NOISE="1"; $env:MIST_MODEL="ppo_mist_robust"; python train_ppo.py
    save_path = MODEL_DIR / MODEL_NAME
    model.save(str(save_path))
    print(f"✅ Training complete. Model saved to: {save_path}.zip")

    robot.simulationQuit(0)


if __name__ == "__main__":
    main()