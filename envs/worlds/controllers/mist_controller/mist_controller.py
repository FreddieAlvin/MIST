from controller import Supervisor
import numpy as np
import math
import sys
import os
import csv
import datetime

# Inject the parent src/ folder path so this agent can locate local utilities if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src')))

# ==========================================
# 1. ARCHITECTURE MODE SWITCH (ENVIRONMENT DRIVEN)
# ==========================================
NAVIGATION_MODE = os.environ.get("MIST_NAV_MODE", "PURE_PPO")
OUTPUT_DIRECTORY = os.environ.get("MIST_OUT_DIR", ".")

CSV_FILENAME = os.path.join(OUTPUT_DIRECTORY, f"{NAVIGATION_MODE}_performance_log.csv")
file_exists = os.path.isfile(CSV_FILENAME)

csv_file = open(CSV_FILENAME, mode='a', newline='', encoding='utf-8')
csv_writer = csv.writer(csv_file)

if not file_exists:
    csv_writer.writerow([
        "Timestamp_Wall_Clock", "Lane_Index", "Navigation_Mode", "Time_Taken_Seconds", "Distance_Traveled_Meters"
    ])
    csv_file.flush()

robot = Supervisor()
timestep = int(robot.getBasicTimeStep())

robot_node = robot.getSelf()
if robot_node is None:
    print("❌ SYSTEM CRITICAL: robot.getSelf() returned None! Ensure supervisor is checked TRUE in the Scene Tree.")
    sys.exit(1)

# ==========================================
# 2. E-PUCK HARDWARE INITIALIZATION DISCOVERY LAYER
# ==========================================
wheels = [robot.getDevice("left wheel motor"), robot.getDevice("right wheel motor")]
for w in wheels:
    w.setPosition(float('inf'))
    w.setVelocity(0.0)

proximity_sensors = []
for i in range(8):
    ps = robot.getDevice(f"ps{i}")
    ps.enable(timestep)
    proximity_sensors.append(ps)

camera = robot.getDevice("camera")
if camera:
    camera.enable(timestep)

START_ARRAY = [
    [-8.8541, 3.13013, 0.0],  # Lane 0
    [-8.8541, 0.370129, 0.0],  # Lane 1
    [-8.8541, -2.36987, 0.0],  # Lane 2
    [-8.8541, -5.12987, 0.0]  # Lane 3
]

GOAL_ARRAY = [
    [10.6659, 3.13013, 0.0],
    [10.6659, 0.370129, 0.0],
    [10.6659, -2.36987, 0.0],
    [10.6659, -5.12987, 0.0]
]

START_ROT = [0, 0, 1, 0]
current_lane_index = 0
num_lanes = len(START_ARRAY)
MAX_SPEED = 6.28  # E-puck maximum motor velocity limit in rad/s
lane_start_sim_time = 0.0
lane_distance_traveled = 0.0
previous_position = None

# ==========================================
# 3. NATIVE EMBEDDED GYMNASIUM TRANSLATION WRAPPER
# ==========================================
import gymnasium as gym
from gymnasium import spaces


class MistNavEnv(gym.Env):
    def __init__(self, robot_supervisor, wheels, proximity_sensors):
        super(MistNavEnv, self).__init__()
        self.robot = robot_supervisor
        self.wheels = wheels
        self.proximity_sensors = proximity_sensors
        self.timestep = int(self.robot.getBasicTimeStep())
        self.robot_node = self.robot.getSelf()

        # Continuous Space: [Forward Velocity, Angular Velocity]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        # Observations: 8 proximity metrics + 2 current wheel velocity states
        self.observation_space = spaces.Box(low=0.0, high=4096.0, shape=(10,), dtype=np.float32)
        self.current_lane = 0

    def _get_obs(self):
        prox_values = np.array([ps.getValue() for ps in self.proximity_sensors], dtype=np.float32)
        v_linear = (self.wheels[0].getVelocity() + self.wheels[1].getVelocity()) / 2.0
        w_angular = (self.wheels[0].getVelocity() - self.wheels[1].getVelocity()) / 0.053
        return np.concatenate([prox_values, [v_linear, w_angular]]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        start_pos = START_ARRAY[self.current_lane]
        self.robot_node.getField("translation").setSFVec3f(start_pos)
        self.robot_node.getField("rotation").setSFRotation(START_ROT)
        self.robot_node.resetPhysics()
        for w in self.wheels:
            w.setVelocity(0.0)
        self.robot.step(self.timestep)
        return self._get_obs(), {}

    def step(self, action):
        base_velocity = float(action[0]) * 4.0
        angular_velocity = float(action[1]) * 2.0

        self.wheels[0].setVelocity(np.clip(base_velocity + angular_velocity, -6.28, 6.28))
        self.wheels[1].setVelocity(np.clip(base_velocity - angular_velocity, -6.28, 6.28))

        self.robot.step(self.timestep)

        obs = self._get_obs()
        pos = self.robot_node.getPosition()

        max_front_prox = np.max([obs[0], obs[1], obs[6], obs[7]])
        reward = 0.2 * base_velocity

        if max_front_prox > 800.0:
            reward -= 2.0

        terminated = False
        if pos[0] >= GOAL_ARRAY[self.current_lane][0]:
            terminated = True
            reward += 100.0
            self.current_lane = (self.current_lane + 1) % 4

        if max_front_prox > 3000.0:
            terminated = True
            reward -= 20.0

        return obs, float(reward), terminated, False, {}


# ==========================================
# 4. ALGORITHMIC NAVIGATION MODELS (D* LITE)
# ==========================================
class DStarLitePlanner:
    def __init__(self, x_bounds=(-10.0, 12.0), y_bounds=(-5.0, 5.0), resolution=0.2):
        self.res = resolution
        self.x_min, self.x_max = x_bounds
        self.y_min, self.y_max = y_bounds
        self.g, self.rhs, self.U = {}, {}, {}
        self.km = 0.0
        self.obstacles = set()
        self.path_waypoints = []
        self.start_cell = None
        self.goal_cell = None

    def w_to_g(self, coord):
        return (int(round((coord[0] - self.x_min) / self.res)), int(round((coord[1] - self.y_min) / self.res)))

    def g_to_w(self, cell):
        return [self.x_min + cell[0] * self.res, self.y_min + cell[1] * self.res]

    def get_neighbors(self, u):
        neighbors = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            v = (u[0] + dx, u[1] + dy)
            if (self.x_min <= self.g_to_w(v)[0] <= self.x_max) and (self.y_min <= self.g_to_w(v)[1] <= self.y_max):
                neighbors.append(v)
        return neighbors

    def h(self, s1, s2):
        return math.hypot(s1[0] - s2[0], s1[1] - s2[1])

    def calculate_key(self, s):
        g_rhs = min(self.g.get(s, float('inf')), self.rhs.get(s, float('inf')))
        return (g_rhs + self.h(self.start_cell, s) + self.km, g_rhs)

    def initialize(self, start_w, goal_w):
        self.g.clear();
        self.rhs.clear();
        self.U.clear();
        self.km = 0.0;
        self.obstacles.clear()
        self.start_cell = self.w_to_g(start_w)
        self.goal_cell = self.w_to_g(goal_w)
        self.rhs[self.goal_cell] = 0.0
        self.U[self.goal_cell] = self.calculate_key(self.goal_cell)

    def cost(self, u, v):
        if u in self.obstacles or v in self.obstacles: return float('inf')
        return self.h(u, v)

    def update_vertex(self, u):
        if u != self.goal_cell:
            self.rhs[u] = min(self.g.get(v, float('inf')) + self.cost(u, v) for v in self.get_neighbors(u))
        if u in self.U: del self.U[u]
        if self.g.get(u, float('inf')) != self.rhs.get(u, float('inf')): self.U[u] = self.calculate_key(u)

    def compute_shortest_path(self):
        while len(self.U) > 0 and (
                min(self.U.values()) < self.calculate_key(self.start_cell) or self.rhs.get(self.start_cell,
                                                                                           float('inf')) != self.g.get(
            self.start_cell, float('inf'))):
            u = min(self.U, key=self.U.get)
            k_old = self.U[u]
            k_new = self.calculate_key(u)
            if k_old < k_new:
                self.U[u] = k_new
            elif self.g.get(u, float('inf')) > self.rhs.get(u, float('inf')):
                self.g[u] = self.rhs[u]
                del self.U[u]
                for v in self.get_neighbors(u): self.update_vertex(v)
            else:
                self.g[u] = float('inf')
                self.update_vertex(u)
                for v in self.get_neighbors(u): self.update_vertex(v)

    def generate_waypoints(self):
        path = []
        curr = self.start_cell
        while curr != self.goal_cell and len(path) < 300:
            path.append(self.g_to_w(curr))
            next_node = min(self.get_neighbors(curr), key=lambda v: self.cost(curr, v) + self.g.get(v, float('inf')),
                            default=None)
            if next_node is None or (self.cost(curr, next_node) + self.g.get(next_node, float('inf'))) == float(
                'inf'): break
            curr = next_node
        path.append(self.g_to_w(self.goal_cell))
        self.path_waypoints = path

    def sense_and_replan(self, current_w, prox_values, current_yaw):
        self.start_cell = self.w_to_g(current_w)
        current_frame_obstacles = set()
        sensor_angles = [0.29, 1.05, 1.57, 2.36, -2.36, -1.57, -1.05, -0.29]

        for idx, val in enumerate(prox_values):
            if val > 600.0:
                dist_est = 0.08 - (val / 4096.0) * 0.06
                global_ang = current_yaw + sensor_angles[idx]
                obs_cell = self.w_to_g(
                    [current_w[0] + dist_est * math.cos(global_ang), current_w[1] + dist_est * math.sin(global_ang)])
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        inflated = (obs_cell[0] + dx, obs_cell[1] + dy)
                        if inflated != self.start_cell and inflated != self.goal_cell: current_frame_obstacles.add(
                            inflated)

        new_obs = current_frame_obstacles - self.obstacles
        cleared_obs = self.obstacles - current_frame_obstacles
        changed = False
        for cell in new_obs:
            self.obstacles.add(cell)
            self.update_vertex(cell)
            for n in self.get_neighbors(cell): self.update_vertex(n)
            changed = True
        for cell in cleared_obs:
            self.obstacles.remove(cell)
            self.update_vertex(cell)
            for n in self.get_neighbors(cell): self.update_vertex(n)
            changed = True
        if changed:
            self.compute_shortest_path()
            self.generate_waypoints()


dstar = DStarLitePlanner()


def reset_to_lane(lane_idx):
    global lane_start_sim_time, lane_distance_traveled, previous_position
    start_pos = START_ARRAY[lane_idx]
    goal_pos = GOAL_ARRAY[lane_idx]
    robot_node.getField("translation").setSFVec3f(start_pos)
    robot_node.getField("rotation").setSFRotation(START_ROT)
    robot_node.resetPhysics()
    for w in wheels: w.setVelocity(0.0)
    if hasattr(reset_to_lane, "is_pivoting"): reset_to_lane.is_pivoting = False
    if robot.getTime() > 0.5 and NAVIGATION_MODE == "PURE_DSTAR":
        dstar.initialize(start_pos[:2], goal_pos[:2])
        dstar.compute_shortest_path()
        dstar.generate_waypoints()
    lane_start_sim_time = robot.getTime()
    lane_distance_traveled = 0.0
    previous_position = np.array([start_pos[0], start_pos[1]])


# ==========================================
# 5. CORE EXECUTION ENGINE CONTROL
# ==========================================
if NAVIGATION_MODE == "PURE_PPO" and not os.environ.get("MIST_EVAL_PPO"):
    # ----------------------------------------------------------------
    # NATIVE REINFORCEMENT LEARNING TRAINING LOOP
    # ----------------------------------------------------------------
    print("⏳ DETECTED PURE_PPO MODE: Spawning Stable-Baselines3 RL Engine directly inside Webots...")
    from stable_baselines3 import PPO

    env = MistNavEnv(robot, wheels, proximity_sensors)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        tensorboard_log=os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "ppo_mist_tensorboard"))
    )

    print("🚀 Initiating Neural Weights Matrix Optimization Loop (40,000 steps)...")
    model.learn(total_timesteps=40000)

    save_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "saved_models"))
    os.makedirs(save_directory, exist_ok=True)
    model.save(os.path.join(save_directory, "ppo_mist_optimal_model"))

    print(f"💾 SUCCESS: Model weights safely archived inside: {save_directory}")
    csv_file.close()
    robot.simulationQuit(0)
    sys.exit(0)

else:
    # ----------------------------------------------------------------
    # ALGORITHMIC MODE & PPO EVALUATION MODE WITH STALL TIMEOUTS
    # ----------------------------------------------------------------
    # --- TIMEOUT GATES ---
    MAX_LANE_DURATION = 90.0  # Absolute maximum seconds allowed per corridor lane
    STALL_CHECK_INTERVAL = 10.0  # Evaluation look-back frequency for deadlocks
    STALL_DISTANCE_THRESHOLD = 0.05  # Robot must cross at least 5cm every 10 seconds

    start_pos = START_ARRAY[current_lane_index]
    robot_node.getField("translation").setSFVec3f(start_pos)
    robot_node.getField("rotation").setSFRotation(START_ROT)
    robot_node.resetPhysics()

    start_time = robot.getTime()
    while robot.step(timestep) != -1:
        if robot.getTime() - start_time >= 1.0: break

    pos_init = robot_node.getPosition()
    goal_pos_init = GOAL_ARRAY[current_lane_index]

    if NAVIGATION_MODE == "PURE_DSTAR":
        dstar.initialize(pos_init[:2], goal_pos_init[:2])
        dstar.compute_shortest_path()
        dstar.generate_waypoints()
    elif NAVIGATION_MODE == "PURE_PPO":
        from stable_baselines3 import PPO

        model_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "saved_models", "ppo_mist_optimal_model.zip"))
        if os.path.exists(model_path):
            model = PPO.load(model_path)
            env = MistNavEnv(robot, wheels, proximity_sensors)
            obs, _ = env.reset()
        else:
            print(f"❌ Model missing at {model_path}! Run training first.")
            robot.simulationQuit(1)
            sys.exit(1)

    lane_start_sim_time = robot.getTime()
    previous_position = np.array([pos_init[0], pos_init[1]])

    last_stall_check_time = robot.getTime()
    last_stall_check_position = np.array([pos_init[0], pos_init[1]])

    K_ATTRACTIVE = float(os.environ.get("MIST_K_ATT", 2.0))
    K_REPULSIVE = float(os.environ.get("MIST_K_REP", 0.05))
    LOOKAHEAD_INDEX = int(os.environ.get("MIST_LOOKAHEAD", 1))
    reset_to_lane.is_pivoting = False

    try:
        while robot.step(timestep) != -1:
            pos = robot_node.getPosition()
            curr_time = robot.getTime()
            current_position_2d = np.array([pos[0], pos[1]])

            if previous_position is not None:
                step_distance = np.linalg.norm(current_position_2d - previous_position)
                if step_distance < 0.5: lane_distance_traveled += step_distance
            previous_position = current_position_2d

            time_spent_in_lane = curr_time - lane_start_sim_time

            # --- GUARD 1: HARD LANE TIMEOUT LIMIT ---
            if time_spent_in_lane > MAX_LANE_DURATION:
                print(
                    f"⚠️ TIMEOUT LIMIT EXCEEDED: Lane {current_lane_index} took > {MAX_LANE_DURATION}s. Terminating run.")
                csv_writer.writerow([
                    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), current_lane_index,
                    f"{NAVIGATION_MODE}_FAILED_TIMEOUT", round(time_spent_in_lane, 3), round(lane_distance_traveled, 3)
                ])
                csv_file.flush()

                current_lane_index = (current_lane_index + 1) % num_lanes
                if current_lane_index == 0:
                    csv_file.close()
                    robot.simulationQuit(0)
                    sys.exit(0)
                reset_to_lane(current_lane_index)
                last_stall_check_time = robot.getTime()
                last_stall_check_position = current_position_2d
                continue

            # --- GUARD 2: POSITION STALL DETECTION (LOCAL MINIMA) ---
            if curr_time - last_stall_check_time >= STALL_CHECK_INTERVAL:
                moved_distance = np.linalg.norm(current_position_2d - last_stall_check_position)
                if moved_distance < STALL_DISTANCE_THRESHOLD:
                    print(
                        f"🛑 STALL/LOCAL MINIMA DETECTED: Moved only {moved_distance:.3f}m in {STALL_CHECK_INTERVAL}s. Skipping lane.")
                    csv_writer.writerow([
                        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), current_lane_index,
                        f"{NAVIGATION_MODE}_LOCAL_MINIMA_STALL", round(time_spent_in_lane, 3),
                        round(lane_distance_traveled, 3)
                    ])
                    csv_file.flush()

                    current_lane_index = (current_lane_index + 1) % num_lanes
                    if current_lane_index == 0:
                        csv_file.close()
                        robot.simulationQuit(0)
                        sys.exit(0)
                    reset_to_lane(current_lane_index)
                    last_stall_check_time = robot.getTime()
                    last_stall_check_position = current_position_2d
                    continue

                last_stall_check_time = curr_time
                last_stall_check_position = current_position_2d

            # --- GOAL ARRIVAL CHECKS ---
            active_goal_x = GOAL_ARRAY[current_lane_index][0]
            if pos[0] >= active_goal_x:
                print(
                    f"🏁 GOAL MET: Lane {current_lane_index} cleared in {time_spent_in_lane:.2f}s | Dist: {lane_distance_traveled:.2f}m")
                csv_writer.writerow([
                    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), current_lane_index, NAVIGATION_MODE,
                    round(time_spent_in_lane, 3), round(lane_distance_traveled, 3)
                ])
                csv_file.flush()

                current_lane_index = (current_lane_index + 1) % num_lanes
                if current_lane_index == 0:
                    print("🎯 BATCH RUN TERMINATION: All lanes complete.")
                    csv_file.close()
                    robot.simulationQuit(0)
                    sys.exit(0)

                reset_to_lane(current_lane_index)
                last_stall_check_time = robot.getTime()
                last_stall_check_position = current_position_2d
                continue

            # --- SENSOR DEPLOYMENT AND MOTION CONTROL ---
            rot_matrix = robot_node.getOrientation()
            current_yaw = math.atan2(rot_matrix[0], rot_matrix[1])
            prox_values = np.array([ps.getValue() for ps in proximity_sensors])

            if NAVIGATION_MODE == "PURE_APF":
                target_pos = np.array([GOAL_ARRAY[current_lane_index][0], GOAL_ARRAY[current_lane_index][1]])
                vec_to_goal = target_pos - current_position_2d
                dist_to_goal = np.linalg.norm(vec_to_goal)
                f_att = K_ATTRACTIVE * (vec_to_goal / dist_to_goal) if dist_to_goal > 0 else np.array([0.0, 0.0])

                f_rep = np.array([0.0, 0.0])
                sensor_angles = [0.29, 1.05, 1.57, 2.36, -2.36, -1.57, -1.05, -0.29]
                for idx, val in enumerate(prox_values):
                    if val > 400.0:
                        global_ray_angle = current_yaw + sensor_angles[idx]
                        vec_away = np.array([-math.cos(global_ray_angle), -math.sin(global_ray_angle)])
                        f_rep += K_REPULSIVE * (val / 4096.0) * vec_away

                f_total = f_att + f_rep
                desired_heading = math.atan2(f_total[1], f_total[0])
                heading_error_rad = (desired_heading - current_yaw + math.pi) % (2 * math.pi) - math.pi

                base_velocity = np.clip(np.linalg.norm(f_total) * 2.0, 1.0, 4.0)
                if abs(heading_error_rad) > 0.5: base_velocity *= 0.2
                angular_velocity = np.clip(heading_error_rad * 3.0, -2.0, 2.0)

            elif NAVIGATION_MODE == "PURE_DSTAR":
                dstar.sense_and_replan(pos[:2], prox_values, current_yaw)
                if len(dstar.path_waypoints) > 0:
                    target_idx = min(LOOKAHEAD_INDEX, len(dstar.path_waypoints) - 1)
                    active_waypoint = dstar.path_waypoints[target_idx]
                else:
                    active_waypoint = [pos[0] + 0.1, GOAL_ARRAY[current_lane_index][1]]

                desired_heading = math.atan2(active_waypoint[1] - pos[1], active_waypoint[0] - pos[0])
                heading_error_rad = (desired_heading - current_yaw + math.pi) % (2 * math.pi) - math.pi

                base_velocity = 3.5
                max_front_prox = max(prox_values[0], prox_values[7])
                if max_front_prox > 1000.0: base_velocity *= 0.2

                if abs(heading_error_rad) > 0.4: reset_to_lane.is_pivoting = True
                if reset_to_lane.is_pivoting:
                    base_velocity = 0.0
                    if abs(heading_error_rad) < 0.1: reset_to_lane.is_pivoting = False
                angular_velocity = np.clip(heading_error_rad * 4.0, -2.5, 2.5)

            elif NAVIGATION_MODE == "PURE_PPO":
                # Evaluation processing logic using the loaded PPO model
                v_linear = (wheels[0].getVelocity() + wheels[1].getVelocity()) / 2.0
                w_angular = (wheels[0].getVelocity() - wheels[1].getVelocity()) / 0.053
                current_obs = np.concatenate([prox_values, [v_linear, w_angular]]).astype(np.float32)

                action, _ = model.predict(current_obs, deterministic=True)
                base_velocity = float(action[0]) * 4.0
                angular_velocity = float(action[1]) * 2.0

            wheels[0].setVelocity(np.clip(base_velocity + angular_velocity, -MAX_SPEED, MAX_SPEED))
            wheels[1].setVelocity(np.clip(base_velocity - angular_velocity, -MAX_SPEED, MAX_SPEED))

    finally:
        csv_file.close()