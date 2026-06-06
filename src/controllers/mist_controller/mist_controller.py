import os
import sys
import math
import csv
import datetime
from pathlib import Path

# ── 1. IMMEDIATE PATH INJECTION (MUST BE FIRST) ──
CONTROLLER_DIR = Path(__file__).resolve().parent
SRC_DIR = CONTROLLER_DIR.parent.parent  # Resolves to MIST/src/
MIST_ROOT = SRC_DIR.parent  # Resolves to MIST/

possible_paths = [SRC_DIR, MIST_ROOT, CONTROLLER_DIR]
for path in possible_paths:
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

# ── 2. WEBOTS ENGINE PLATFORM PATH COMPLIANCE ──
if sys.platform == "darwin":
    os.environ['WEBOTS_HOME'] = '/Applications/Webots.app'
elif sys.platform == "win32":
    os.environ['WEBOTS_HOME'] = 'C:\\Program Files\\Webots'
else:
    os.environ['WEBOTS_HOME'] = '/usr/local/webots'

webots_path = os.path.join(os.environ['WEBOTS_HOME'], 'Contents', 'lib', 'controller',
                           'python') if sys.platform == "darwin" else \
    os.path.join(os.environ['WEBOTS_HOME'], 'lib', 'controller', 'python')

if os.path.exists(webots_path):
    if webots_path not in sys.path:
        sys.path.append(webots_path)

from controller import Supervisor
import numpy as np

# ── 3. TESTING CONTROL TOGGLES ──
DEBUG_HOLD_LANE = True

# ── 4. LOGGING & FILE MANAGEMENT ──
LOG_DIR = MIST_ROOT / "results" / "logs"
MODEL_DIR = MIST_ROOT / "saved_models"
LOG_DIR.mkdir(parents=True, exist_ok=True)

NAVIGATION_MODE = os.environ.get("MIST_NAV_MODE", "PURE_DSTAR")
CSV_PATH = LOG_DIR / f"{NAVIGATION_MODE}_performance_log.csv"
file_exists = CSV_PATH.exists()

csv_file = open(str(CSV_PATH), mode='a', newline='', encoding='utf-8')
csv_writer = csv.writer(csv_file)

if not file_exists:
    csv_writer.writerow([
        "Timestamp_Wall_Clock", "Lane_Index", "Navigation_Mode",
        "Time_Taken_Seconds", "Distance_Traveled_Meters"
    ])
    csv_file.flush()

# ── 5. CORE INITIALIZATION & PRIVILEGE CHECKS ──
robot = Supervisor()
timestep = int(robot.getBasicTimeStep())

robot_node = robot.getSelf()
if robot_node is None:
    print("❌ SYSTEM CRITICAL: robot.getSelf() returned None! Grant supervisor privileges.")
    sys.exit(1)

# ── 6. AUTOMATED HARDWARE DISCOVERY LAYER (EXACTLY MATCHED) ──
lidar = None
camera = None
discovered_motors = []
device_count = robot.getNumberOfDevices()

for i in range(device_count):
    device = robot.getDeviceByIndex(i)
    class_name = device.__class__.__name__

    if class_name == "Lidar" and lidar is None:
        lidar = device
    elif class_name == "Camera" and camera is None:
        camera = device
    elif class_name == "Motor":
        discovered_motors.append(device)

if lidar:
    lidar.enable(timestep)
else:
    print("❌ SYSTEM CRITICAL: Scanning LiDAR sensor could not be found anywhere on this robot!")
    sys.exit(1)

if camera:
    camera.enable(timestep)

wheels = [None, None]
for motor in discovered_motors:
    motor_name = motor.getName().lower()
    if "left" in motor_name or "m1" in motor_name:
        wheels[0] = motor
    elif "right" in motor_name or "m2" in motor_name:
        wheels[1] = motor

if wheels[0] is None or wheels[1] is None:
    if len(discovered_motors) >= 2:
        wheels = [discovered_motors[0], discovered_motors[1]]
    else:
        print("❌ SYSTEM CRITICAL: Differential wheel motors missing!")
        sys.exit(1)

for w in wheels:
    w.setPosition(float('inf'))
    w.setVelocity(0.0)

# ── 7. ENVIRONMENT & KINEMATIC MATRIX SETTINGS ──
START_ARRAY = [
    [-8.8541, 3.13013, -0.000217358],  # Lane 0
    [-8.8541, 0.370129, -0.000217358],  # Lane 1
    [-8.8541, -2.36987, -0.000217358]   # Lane 2
]

GOAL_ARRAY = [
    [10.6659, 3.13013, -0.000217358],  # Lane 0 Exit Line
    [10.6659, 0.370129, -0.000217358],  # Lane 1 Exit Line
    [10.6659, -2.36987, -0.000217358]   # Lane 2 Exit Line
]

START_ROT = [0, 0, 1, 0]
current_lane_index = 0
num_lanes = len(START_ARRAY)
MAX_SPEED = 6.28
lane_start_sim_time = 0.0
lane_distance_traveled = 0.0
previous_position = None

last_stall_check_time = 0.0
last_stall_check_position = np.array([0.0, 0.0])


# ── 8. D* LITES PATH ENGINE MODULE (LIDAR REFACTOR) ──
class DStarLitePlanner:
    def __init__(self, x_bounds=(-10.0, 12.0), y_bounds=(-5.0, 5.0), resolution=0.15):
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
        self.g.clear()
        self.rhs.clear()
        self.U.clear()
        self.km = 0.0
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
        while curr != self.goal_cell and len(path) < 400:
            path.append(self.g_to_w(curr))
            next_node = min(self.get_neighbors(curr), key=lambda v: self.cost(curr, v) + self.g.get(v, float('inf')),
                            default=None)
            if next_node is None or (self.cost(curr, next_node) + self.g.get(next_node, float('inf'))) == float('inf'):
                break
            curr = next_node
        path.append(self.g_to_w(self.goal_cell))
        self.path_waypoints = path

    def sense_and_replan(self, current_w, horizon, current_yaw, H_RES, DEG_PER_INDEX):
        self.start_cell = self.w_to_g(current_w)
        changed_detected = False
        half_res = H_RES // 2

        for idx in range(H_RES):
            dist = horizon[idx]
            if dist < 0.40:
                if idx <= half_res:
                    ray_angle = math.radians(idx * DEG_PER_INDEX)
                else:
                    ray_angle = math.radians((idx - H_RES) * DEG_PER_INDEX)

                global_ang = current_yaw + ray_angle
                ox = current_w[0] + dist * math.cos(global_ang)
                oy = current_w[1] + dist * math.sin(global_ang)
                obs_cell = self.w_to_g([ox, oy])

                if obs_cell not in self.obstacles and obs_cell != self.start_cell and obs_cell != self.goal_cell:
                    # --- CONFIGURATION: 3x3 RADIAL GRID INFLATION RADIUS ---
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            inflated_cell = (obs_cell[0] + dx, obs_cell[1] + dy)
                            if inflated_cell != self.start_cell and inflated_cell != self.goal_cell:
                                self.obstacles.add(inflated_cell)
                                for n in self.get_neighbors(inflated_cell):
                                    self.update_vertex(n)
                                self.update_vertex(inflated_cell)
                    changed_detected = True

        if changed_detected:
            print("🧱 D* LITE MAP UPDATE: Map Node Change Registered! Recalculating Matrix Paths...")
            self.compute_shortest_path()
            self.generate_waypoints()


dstar = DStarLitePlanner()


def reset_to_lane(lane_idx):
    global lane_start_sim_time, lane_distance_traveled, previous_position
    global last_stall_check_time, last_stall_check_position

    start_pos = START_ARRAY[lane_idx]
    goal_pos = GOAL_ARRAY[lane_idx]
    print(f"🚀 TELEPORTING ROBOT TO LANE {lane_idx} -> Coordinates: {start_pos}")
    robot_node.getField("translation").setSFVec3f(start_pos)
    robot_node.getField("rotation").setSFRotation(START_ROT)
    robot_node.resetPhysics()
    for w in wheels: w.setVelocity(0.0)

    if robot.getTime() > 0.5 and NAVIGATION_MODE == "PURE_DSTAR":
        dstar.initialize(start_pos[:2], goal_pos[:2])
        dstar.compute_shortest_path()
        dstar.generate_waypoints()

    lane_start_sim_time = robot.getTime()
    lane_distance_traveled = 0.0
    previous_position = np.array([start_pos[0], start_pos[1]])

    last_stall_check_time = robot.getTime()
    last_stall_check_position = np.array([start_pos[0], start_pos[1]])


# --- FIX: HARD 10-SECOND POSITION OR ONE-CELL STALL THRESHOLD MATRIX ---
# Disables global countdown timeouts. Robot runs until it makes progress or explicitly stalls.
STALL_DISTANCE_THRESHOLD = 0.15  # Equivalent to 1 full resolution cell width
STALL_CHECK_INTERVAL = 10.0      # Evaluation timeframe limit

reset_to_lane(current_lane_index)

start_time = robot.getTime()
while robot.step(timestep) != -1:
    if robot.getTime() - start_time >= 1.0:
        break

H_RES = lidar.getHorizontalResolution()
half_res = H_RES // 2
DEG_PER_INDEX = 360.0 / H_RES

K_ATTRACTIVE = 2.5
K_REPULSIVE = 0.08
INFLUENCE_DIST = 0.45
WINDOW_SIZE = 2

last_metric_print_time = 0.0

print(f"🚀 Started Evaluator Suite. Native Hardware Engine active: {NAVIGATION_MODE}")

# ── 9. MAIN NAVIGATION CONTROLLER LOOP ──
try:
    while robot.step(timestep) != -1:
        pos = robot_node.getPosition()
        curr_time = robot.getTime()
        current_position_2d = np.array([pos[0], pos[1]])

        if previous_position is not None:
            step_distance = np.linalg.norm(current_position_2d - previous_position)
            if step_distance < 0.5:
                lane_distance_traveled += step_distance
        previous_position = current_position_2d

        time_spent_in_lane = curr_time - lane_start_sim_time

        if camera:
            camera.getImage()


        def advance_or_hold_lane(log_status):
            global current_lane_index
            csv_writer.writerow([
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), current_lane_index,
                log_status, round(time_spent_in_lane, 3), round(lane_distance_traveled, 3)
            ])
            csv_file.flush()

            if DEBUG_HOLD_LANE:
                print(f"🔄 DEBUG RETRY: Re-running Lane {current_lane_index}.")
            else:
                current_lane_index = (current_lane_index + 1) % num_lanes
            reset_to_lane(current_lane_index)


        # --- STALL ENGINE ---
        # Triggered only if the robot does not advance out of its resolution box every 10 seconds.
        if curr_time - last_stall_check_time >= STALL_CHECK_INTERVAL:
            moved_distance = np.linalg.norm(current_position_2d - last_stall_check_position)
            if moved_distance < STALL_DISTANCE_THRESHOLD:
                print(f"🛑 STALL DETECTED: Position locked inside 1-square space over 10s ({moved_distance:.3f}m moved).")
                advance_or_hold_lane(f"{NAVIGATION_MODE}_LOCAL_MINIMA_STALL")
                if current_lane_index == 0 and not DEBUG_HOLD_LANE: break
                continue
            last_stall_check_time = curr_time
            last_stall_check_position = current_position_2d

        active_goal_x = GOAL_ARRAY[current_lane_index][0]
        if pos[0] >= active_goal_x:
            print(f"🏁 GOAL MET: Lane {current_lane_index} cleared in {time_spent_in_lane:.2f}s")
            csv_writer.writerow([
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), current_lane_index, NAVIGATION_MODE,
                round(time_spent_in_lane, 3), round(lane_distance_traveled, 3)
            ])
            csv_file.flush()

            current_lane_index = (current_lane_index + 1) % num_lanes
            if current_lane_index == 0:
                print("🎯 BATCH RUN TERMINATION: All lanes complete.")
                break
            reset_to_lane(current_lane_index)
            continue

        # CORRECT COMPASS MATRIX TRANSFORMATION
        rot_matrix = robot_node.getOrientation()
        current_yaw = math.atan2(rot_matrix[3], rot_matrix[0]) - (math.pi / 2.0)
        current_yaw = (current_yaw + math.pi) % (2 * math.pi) - math.pi

        lidar_raw = lidar.getRangeImage()
        if lidar_raw is None:
            continue
        horizon = np.array(lidar_raw)
        horizon[np.isinf(horizon)] = 3.0
        horizon[horizon <= 0.05] = 3.0

        if NAVIGATION_MODE == "PURE_APF":
            # ============================================================
            # MODEL 1: PURE ARTIFICIAL POTENTIAL FIELDS CONTROLLER
            # ============================================================
            target_pos = np.array([GOAL_ARRAY[current_lane_index][0], GOAL_ARRAY[current_lane_index][1]])
            vec_to_goal = target_pos - current_position_2d
            dist_to_goal = np.linalg.norm(vec_to_goal)
            f_att = K_ATTRACTIVE * (vec_to_goal / dist_to_goal) if dist_to_goal > 0 else np.array([0.0, 0.0])

            f_rep = np.array([0.0, 0.0])
            for idx in range(H_RES):
                dist_reading = horizon[idx]

                is_local_minimum = True
                for offset in range(-WINDOW_SIZE, WINDOW_SIZE + 1):
                    neighbor_idx = (idx + offset) % H_RES
                    if horizon[neighbor_idx] < dist_reading:
                        is_local_minimum = False
                        break

                if not is_local_minimum:
                    continue

                if dist_reading < INFLUENCE_DIST and dist_reading > 0.01:
                    if idx <= half_res:
                        ray_angle_rad = math.radians(idx * DEG_PER_INDEX)
                    else:
                        ray_angle_rad = math.radians((idx - H_RES) * DEG_PER_INDEX)

                    global_ray_angle = current_yaw + ray_angle_rad
                    vec_away_from_wall = np.array([-math.cos(global_ray_angle), -math.sin(global_ray_angle)])

                    factor = (1.0 / dist_reading) - (1.0 / INFLUENCE_DIST)
                    magnitude = K_REPULSIVE * factor * (1.0 / (dist_reading ** 2))
                    f_rep += magnitude * vec_away_from_wall

            f_total = f_att + f_rep

            desired_heading = math.atan2(f_total[1], f_total[0])
            heading_error_rad = (desired_heading - current_yaw + math.pi) % (2 * math.pi) - math.pi

            force_magnitude = np.linalg.norm(f_total)
            base_velocity = np.clip(force_magnitude * 1.2, 1.0, 4.0)
            if abs(heading_error_rad) > 0.6:
                base_velocity *= 0.3

            angular_velocity = np.clip(heading_error_rad * 3.5, -2.5, 2.5)

        elif NAVIGATION_MODE == "PURE_DSTAR":
            # ============================================================
            # MODEL 2: OPTIMIZED PURE D* LITE GEOMETRIC CONTROLLER
            # ============================================================
            dstar.sense_and_replan(pos[:2], horizon, current_yaw, H_RES, DEG_PER_INDEX)

            if len(dstar.path_waypoints) > 0:
                active_waypoint = dstar.path_waypoints[0]
                dist_to_waypoint = math.hypot(active_waypoint[0] - pos[0], active_waypoint[1] - pos[1])

                if dist_to_waypoint < 0.26 and len(dstar.path_waypoints) > 1:
                    dstar.path_waypoints.pop(0)
                    active_waypoint = dstar.path_waypoints[0]
                    dist_to_waypoint = math.hypot(active_waypoint[0] - pos[0], active_waypoint[1] - pos[1])
            else:
                active_waypoint = [GOAL_ARRAY[current_lane_index][0], GOAL_ARRAY[current_lane_index][1]]
                dist_to_waypoint = math.hypot(active_waypoint[0] - pos[0], active_waypoint[1] - pos[1])

            desired_heading = math.atan2(active_waypoint[1] - pos[1], active_waypoint[0] - pos[0])
            heading_error_rad = (desired_heading - current_yaw + math.pi) % (2 * math.pi) - math.pi

            front_center_idx = H_RES // 2
            min_front_wall_distance = min(horizon[max(0, front_center_idx - 4): min(H_RES, front_center_idx + 5)])

            base_velocity = 3.0

            if dist_to_waypoint < 0.40:
                base_velocity = max(0.5, base_velocity * (dist_to_waypoint / 0.40))

            if min_front_wall_distance < 0.40:
                base_velocity *= (min_front_wall_distance / 0.40)

            if abs(heading_error_rad) > 0.20:
                base_velocity = 0.0

            angular_velocity = np.clip(heading_error_rad * 7.5, -4.0, 4.0)

        # Uniform differential kinematic mixer matching evaluation environments
        wheels[0].setVelocity(np.clip(base_velocity - angular_velocity, -MAX_SPEED, MAX_SPEED))
        wheels[1].setVelocity(np.clip(base_velocity + angular_velocity, -MAX_SPEED, MAX_SPEED))

except Exception as e:
    print(f"❌ CRITICAL RUNTIME EXCEPTION: {e}")
    raise e

finally:
    print("💾 Closing performance data streams.")
    csv_file.close()
    robot.simulationQuit(0)