"""
run_grid_search.py  –  Evaluates the trained PPO model across three fog noise
levels and records per-lane performance metrics.

Must be run as a Webots extern controller (the robot node must be set to
controller "<extern>").
"""

import os
import sys
import csv
import datetime
from pathlib import Path

from controller import Supervisor

# ── Path resolution ───────────────────────────────────────────────────────────
# Get the directory where this script (run_grid_search.py) is located
SRC_DIR = Path(__file__).resolve().parent
# Define MIST_ROOT based on the location of the src folder
MIST_ROOT = SRC_DIR.parent

# Add 'src/utils' to the path explicitly so we can import mist_env
sys.path.insert(0, str(SRC_DIR / "utils"))

from mist_env import MistNavEnv

# ── Stable-Baselines3 import ──────────────────────────────────────────────────
try:
    from stable_baselines3 import PPO
except ImportError as exc:
    print("❌ stable_baselines3 not found. Install it with: pip install stable-baselines3")
    raise exc

# ── Configuration ─────────────────────────────────────────────────────────────
NOISE_LEVELS = [0.0, 0.01, 0.02]   # normalised prox noise std-devs
NUM_RUNS_PER_NOISE = 4              # one run = one full 4-lane pass

MODEL_PATH = Path(
    os.environ.get(
        "MIST_MODEL_PATH",
        str(MIST_ROOT / "saved_models" / "ppo_mist_optimal_model")
    )
).with_suffix(".zip")

OUT_DIR  = Path(os.environ.get("MIST_OUT_DIR", "."))
CSV_PATH = OUT_DIR / "grid_search_results.csv"


def main():
    # ── Webots hardware ───────────────────────────────────────────────────────
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

    # ── Load model ────────────────────────────────────────────────────────────
    if not MODEL_PATH.exists():
        print(f"❌ Model not found at {MODEL_PATH}")
        print("   Run train_ppo.py first to generate the model.")
        robot.simulationQuit(1)
        return

    print(f"✅ Loading model from {MODEL_PATH}")
    model = PPO.load(str(MODEL_PATH))

    # ── CSV output ────────────────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    new_file = not CSV_PATH.exists()
    csv_fh   = open(CSV_PATH, mode='a', newline='', encoding='utf-8')
    writer   = csv.writer(csv_fh)
    if new_file:
        writer.writerow([
            "Timestamp", "Noise_Level", "Run_Index", "Lane_Index",
            "Status", "Time_Seconds", "Steps", "Goal_Reached"
        ])
        csv_fh.flush()

    # ── Grid search loop ──────────────────────────────────────────────────────
    for noise in NOISE_LEVELS:
        print(f"\n{'─'*60}")
        print(f"  Noise level: {noise:.3f}  ({NUM_RUNS_PER_NOISE} runs)")
        print(f"{'─'*60}")

        for run_idx in range(NUM_RUNS_PER_NOISE):
            env = MistNavEnv(robot, wheels, proximity_sensors,
                             noise_level=noise)
            obs, _ = env.reset()

            lane_start_time = robot.getTime()
            lane_idx        = env.current_lane
            total_steps     = 0
            done            = False

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                total_steps += 1
                done = terminated or truncated

                if env.current_lane != lane_idx:
                    elapsed = robot.getTime() - lane_start_time
                    status  = "SUCCESS" if terminated else "TRUNCATED"
                    print(f"    Run {run_idx} | Lane {lane_idx} | "
                          f"Noise={noise:.2f} | {status} | {elapsed:.2f}s | {total_steps} steps")
                    writer.writerow([
                        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        noise, run_idx, lane_idx, status,
                        round(elapsed, 3), total_steps, terminated
                    ])
                    csv_fh.flush()
                    lane_idx        = env.current_lane
                    lane_start_time = robot.getTime()
                    total_steps     = 0

    csv_fh.close()
    print(f"\n🎯 Grid search complete. Results saved to {CSV_PATH}")
    robot.simulationQuit(0)


if __name__ == "__main__":
    main()