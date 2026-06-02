from controller import Supervisor
import numpy as np
import math
import sys

# ==========================================
# 1. ARCHITECTURE MODE SWITCH
# ==========================================
NAVIGATION_MODE = "PURE_DSTAR"

# ==========================================
# 2. CORE INITIALIZATION & PRIVILEGE CHECKS
# ==========================================
robot = Supervisor()
timestep = int(robot.getBasicTimeStep())

robot_node = robot.getSelf()
if robot_node is None:
    print("❌ SYSTEM CRITICAL: robot.getSelf() returned None! Grant supervisor privileges.")
    sys.exit(1)

# ==========================================
# 3. AUTOMATED HARDWARE DISCOVERY LAYER
# ==========================================
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

# ==========================================
# 4. ENVIRONMENT & KINEMATIC MATRIX SETTINGS
# ==========================================
START_ARRAY = [
    [-8.8541, 3.13013, -0.000217358],  # Lane 0
    [-8.8541, 0.370129, -0.000217358],  # Lane 1
    [-8.8541, -2.36987, -0.000217358]  # Lane 2
]

GOAL_ARRAY = [
    [10.6659, 3.13013, -0.000217358],  # Lane 0 Exit Line
    [10.6659, 0.370129, -0.000217358],  # Lane 1 Exit Line
    [10.6659, -2.36987, -0.000217358]  # Lane 2 Exit Line
]

START_ROT = [0, 0, 1, 0]
current_lane_index = 0
num_lanes = len(START_ARRAY)

MAX_SPEED = 6.28


# ==========================================
# 5. ALGORITHM MODEL A: STANDALONE D* LITE ENGINE
# ==========================================
class DStarLitePlanner:
    def __init__(self, x_bounds=(-10.0, 12.0), y_bounds=(-5.0, 5.0), resolution=0.15):
        self.res = resolution
        self.x_min, self.x_max = x_bounds
        self.y_min, self.y_max = y_bounds

        self.g = {}
        self.rhs = {}
        self.U = {}
        self.km = 0.0
        self.obstacles = set()

        self.path_waypoints = []
        self.start_cell = None
        self.goal_cell = None

    def w_to_g(self, coord):
        gx = int(round((coord[0] - self.x_min) / self.res))
        gy = int(round((coord[1] - self.y_min) / self.res))
        return (gx, gy)

    def g_to_w(self, cell):
        wx = self.x_min + cell[0] * self.res
        wy = self.y_min + cell[1] * self.res
        return [wx, wy]

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
        k1 = g_rhs + self.h(self.start_cell, s) + self.km
        k2 = g_rhs
        return (k1, k2)

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
        if u in self.obstacles or v in self.obstacles:
            return float('inf')
        return self.h(u, v)

    def update_vertex(self, u):
        if u != self.goal_cell:
            min_rhs = float('inf')
            for v in self.get_neighbors(u):
                min_rhs = min(min_rhs, self.g.get(v, float('inf')) + self.cost(u, v))
            self.rhs[u] = min_rhs

        if u in self.U:
            del self.U[u]

        if self.g.get(u, float('inf')) != self.rhs.get(u, float('inf')):
            self.U[u] = self.calculate_key(u)

    def compute_shortest_path(self):
        while len(self.U) > 0 and (min(self.U.values()) < self.calculate_key(self.start_cell) or
                                   self.rhs.get(self.start_cell, float('inf')) != self.g.get(self.start_cell,
                                                                                             float('inf'))):
            u = min(self.U, key=self.U.get)
            k_old = self.U[u]
            k_new = self.calculate_key(u)

            if k_old < k_new:
                self.U[u] = k_new
            elif self.g.get(u, float('inf')) > self.rhs.get(u, float('inf')):
                self.g[u] = self.rhs[u]
                del self.U[u]
                for v in self.get_neighbors(u):
                    self.update_vertex(v)
            else:
                self.g[u] = float('inf')
                self.update_vertex(u)
                for v in self.get_neighbors(u):
                    self.update_vertex(v)

    def generate_waypoints(self):
        path = []
        curr = self.start_cell
        max_steps = 400
        steps = 0

        while curr != self.goal_cell and steps < max_steps:
            path.append(self.g_to_w(curr))
            min_cost = float('inf')
            next_node = None
            for v in self.get_neighbors(curr):
                move_cost = self.cost(curr, v) + self.g.get(v, float('inf'))
                if move_cost < min_cost:
                    min_cost = move_cost
                    next_node = v
            if next_node is None or min_cost == float('inf'):
                break
            curr = next_node
            steps += 1

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
                    for dx in [-2, -1, 0, 1, 2]:
                        for dy in [-2, -1, 0, 1, 2]:
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
    """Teleports the e-puck to the beginning of the selected lane."""
    start_pos = START_ARRAY[lane_idx]
    goal_pos = GOAL_ARRAY[lane_idx]
    print(f"🚀 TELEPORTING E-PUCK TO LANE {lane_idx} -> Coordinates: {start_pos}")
    robot_node.getField("translation").setSFVec3f(start_pos)
    robot_node.getField("rotation").setSFRotation(START_ROT)
    robot_node.resetPhysics()
    for w in wheels:
        w.setVelocity(0.0)

    if NAVIGATION_MODE == "PURE_DSTAR":
        dstar.initialize(start_pos[:2], goal_pos[:2])
        dstar.compute_shortest_path()
        dstar.generate_waypoints()


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

# ==========================================
# 6. MAIN NAVIGATION CONTROLLER LOOP
# ==========================================
while robot.step(timestep) != -1:
    pos = robot_node.getPosition()
    curr_time = robot.getTime()

    if camera:
        camera.getImage()

    active_goal_x = GOAL_ARRAY[current_lane_index][0]
    if pos[0] >= active_goal_x:
        print(f"🏁 GOAL MET: Lane {current_lane_index} cleared successfully!")
        current_lane_index = (current_lane_index + 1) % num_lanes
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
        robot_pos = np.array([pos[0], pos[1]])

        vec_to_goal = target_pos - robot_pos
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

            # --- FIX 1: INCREASE ACCEPTANCE ENVELOPE TO 0.26m ---
            if dist_to_waypoint < 0.26 and len(dstar.path_waypoints) > 1:
                dstar.path_waypoints.pop(0)
                active_waypoint = dstar.path_waypoints[0]
                dist_to_waypoint = math.hypot(active_waypoint[0] - pos[0], active_waypoint[1] - pos[1])
        else:
            active_waypoint = [GOAL_ARRAY[current_lane_index][0], GOAL_ARRAY[current_lane_index][1]]
            dist_to_waypoint = math.hypot(active_waypoint[0] - pos[0], active_waypoint[1] - pos[1])

        desired_heading = math.atan2(active_waypoint[1] - pos[1], active_waypoint[0] - pos[0])
        heading_error_rad = (desired_heading - current_yaw + math.pi) % (2 * math.pi) - math.pi

        # EMERGENCY LIDAR BRAKING MATRIX
        front_center_idx = H_RES // 2
        min_front_wall_distance = min(horizon[max(0, front_center_idx - 4): min(H_RES, front_center_idx + 5)])

        # Base translation forward speed
        base_velocity = 3.0

        if dist_to_waypoint < 0.40:
            base_velocity = max(0.5, base_velocity * (dist_to_waypoint / 0.40))

        if min_front_wall_distance < 0.40:
            base_velocity *= (min_front_wall_distance / 0.40)

        # --- FIX 2: ZERO-VELOCITY SPOT TURN MATRIX ---
        # If alignment error is noticeable, do not crawl forward at all; spin in place.
        if abs(heading_error_rad) > 0.20:
            base_velocity = 0.0

            # Increased angular steering sensitivity for rapid rotation snapping
        angular_velocity = np.clip(heading_error_rad * 7.5, -4.0, 4.0)

    # ----------------------------------------------------------------
    # DIFFERENTIAL DRIVE KINEMATIC MIXER
    # ----------------------------------------------------------------
    v_l = base_velocity - angular_velocity
    v_r = base_velocity + angular_velocity

    wheels[0].setVelocity(np.clip(v_l, -MAX_SPEED, MAX_SPEED))
    wheels[1].setVelocity(np.clip(v_r, -MAX_SPEED, MAX_SPEED))

    # ----------------------------------------------------------------
    # DIAGNOSTIC PROFILE CONSOLE DASHBOARD
    # ----------------------------------------------------------------
    if curr_time - last_metric_print_time >= 1.5:
        last_metric_print_time = curr_time
        print("=" * 65)
        print(f"📊 ISOLATED SYSTEM MONITOR LOGS [Time: {curr_time:.2f}s]")
        print(f"⚙️ Running Model Variant : {NAVIGATION_MODE}")
        print(
            f"📍 Robot Coordinates     : X={pos[0]:.2f}, Y={pos[1]:.2f} | Aligned Heading: {math.degrees(current_yaw):.1f}°")
        if NAVIGATION_MODE == "PURE_DSTAR":
            print(
                f"⛓️  Active D* Target Node : X={active_waypoint[0]:.2f}, Y={active_waypoint[1]:.2f} | Distance: {dist_to_waypoint:.2f}m | Nodes Left: {len(dstar.path_waypoints)}")
            print(f"🚨 Proportional Braking  : Front Clear Buffer Window = {min_front_wall_distance:.2f}m")
        else:
            print(f"🚀 APF Net Target Vector  : X-Force={f_total[0]:.2f}, Y-Force={f_total[1]:.2f}")
        print(f"⚡ Motor Thrust Metrics  : Left={v_l:.2f} rad/s | Right={v_r:.2f} rad/s")
        print("=" * 65)