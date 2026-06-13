import os
import sys
import math
import csv
import datetime
from pathlib import Path

# ── 1. IMMEDIATE PATH INJECTION (MUST BE FIRST) ──────────────────────────────
CONTROLLER_DIR = Path(__file__).resolve().parent
SRC_DIR  = CONTROLLER_DIR.parent.parent   # MIST/src/
MIST_ROOT = SRC_DIR.parent                # MIST/

for path in [SRC_DIR, MIST_ROOT, CONTROLLER_DIR]:
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

# ── 2. WEBOTS ENGINE PLATFORM PATH COMPLIANCE ─────────────────────────────────
if sys.platform == "darwin":
    os.environ['WEBOTS_HOME'] = '/Applications/Webots.app'
elif sys.platform == "win32":
    os.environ['WEBOTS_HOME'] = 'C:\\Program Files\\Webots'
else:
    os.environ['WEBOTS_HOME'] = '/usr/local/webots'

webots_path = (
    os.path.join(os.environ['WEBOTS_HOME'], 'Contents', 'lib', 'controller', 'python')
    if sys.platform == "darwin"
    else os.path.join(os.environ['WEBOTS_HOME'], 'lib', 'controller', 'python')
)
if os.path.exists(webots_path) and webots_path not in sys.path:
    sys.path.append(webots_path)

from controller import Supervisor
import numpy as np

# ── 3. TESTING CONTROL TOGGLES ────────────────────────────────────────────────
# Set to False for a full unattended run across all lanes.
DEBUG_HOLD_LANE = False

# ── 4. LOGGING & FILE MANAGEMENT ─────────────────────────────────────────────
LOG_DIR   = MIST_ROOT / "results" / "logs"
MODEL_DIR = MIST_ROOT / "saved_models"
LOG_DIR.mkdir(parents=True, exist_ok=True)

NAVIGATION_MODE = os.environ.get("MIST_NAV_MODE", "PURE_DSTAR")
CSV_PATH    = LOG_DIR / f"{NAVIGATION_MODE}_performance_log.csv"
file_exists = CSV_PATH.exists()

csv_file   = open(str(CSV_PATH), mode='a', newline='', encoding='utf-8')
csv_writer = csv.writer(csv_file)
if not file_exists:
    csv_writer.writerow([
        "Timestamp_Wall_Clock", "Lane_Index", "Navigation_Mode",
        "Time_Taken_Seconds",   "Distance_Traveled_Meters"
    ])
    csv_file.flush()

# ── 5. CORE INITIALISATION & PRIVILEGE CHECKS ─────────────────────────────────
robot    = Supervisor()
timestep = int(robot.getBasicTimeStep())

robot_node = robot.getSelf()
if robot_node is None:
    print("❌ SYSTEM CRITICAL: robot.getSelf() returned None! Grant supervisor privileges.")
    sys.exit(1)

# ── 6. HARDWARE DISCOVERY ─────────────────────────────────────────────────────
#
# FIX – BUG 1 (CRASH): The e-puck has NO Lidar device.  The original controller
# called sys.exit(1) the moment it could not find one.  The e-puck's sensing is
# provided by 8 infrared proximity sensors (ps0..ps7).  We discover wheels the
# same way as before, but replace the lidar block entirely with ps initialisation.
#
wheels = [None, None]
discovered_motors = []
device_count = robot.getNumberOfDevices()
for i in range(device_count):
    device    = robot.getDeviceByIndex(i)
    cls_name  = device.__class__.__name__
    if cls_name == "Motor":
        discovered_motors.append(device)

for motor in discovered_motors:
    name = motor.getName().lower()
    if "left"  in name or "m1" in name:
        wheels[0] = motor
    elif "right" in name or "m2" in name:
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

# Initialise the 8 proximity sensors that replace the lidar.
# ps0 (front-right, +10°) .. ps7 (front-left, -10°).
proximity_sensors = []
for i in range(8):
    ps = robot.getDevice(f"ps{i}")
    if ps is None:
        print(f"❌ SYSTEM CRITICAL: Proximity sensor ps{i} not found!")
        sys.exit(1)
    ps.enable(timestep)
    proximity_sensors.append(ps)

# Angles of ps0..ps7 in the robot's own frame (radians, CCW positive).
# Source: GCtronic e-puck hardware specification.
PS_ANGLES = [
    math.radians( 10),   # ps0 – front-right
    math.radians( 45),   # ps1 – right-front diagonal
    math.radians( 90),   # ps2 – right
    math.radians(150),   # ps3 – rear-right
    math.radians(-150),  # ps4 – rear-left
    math.radians(-90),   # ps5 – left
    math.radians(-45),   # ps6 – left-front diagonal
    math.radians(-10),   # ps7 – front-left
]

# Maximum useful range of the e-puck IR sensors (~10 cm).
PS_MAX_RANGE = 0.10   # metres

def ps_to_dist(reading: float) -> float:
    """Convert a raw ADC reading [0..4096] to an estimated distance in metres.
    Linear approximation: 0 → 0.10 m (far), 4096 → 0.01 m (contact).
    """
    return max(0.01, PS_MAX_RANGE * (1.0 - min(reading / 4096.0, 1.0)))

camera = robot.getDevice("camera")
if camera:
    camera.enable(timestep)

# ── 7. ENVIRONMENT & KINEMATIC MATRIX SETTINGS ───────────────────────────────
START_ARRAY = [
    [-8.8541,  3.13013, -0.000217358],   # Lane 0
    [-8.8541,  0.370129, -0.000217358],  # Lane 1
    [-8.8541, -2.36987, -0.000217358],   # Lane 2
]
GOAL_ARRAY = [
    [10.6659,  3.13013, -0.000217358],
    [10.6659,  0.370129, -0.000217358],
    [10.6659, -2.36987, -0.000217358],
]
START_ROT = [0, 0, 1, 0]

current_lane_index   = 0
num_lanes            = len(START_ARRAY)
MAX_SPEED            = 6.28      # e-puck motor limit (rad/s)
lane_start_sim_time  = 0.0
lane_distance_traveled = 0.0
previous_position    = None

last_stall_check_time     = 0.0
last_stall_check_position = np.array([0.0, 0.0])

# ── 8. D* LITE PATH ENGINE ────────────────────────────────────────────────────
class DStarLitePlanner:
    """
    Incremental D* Lite planner on a 2-D occupancy grid.

    Fixes applied vs. the original:
      • BUG 5 (STALE MAP): sense_and_replan() now maintains a
        `_prev_frame_obstacles` set and symmetrically REMOVES cells that
        were detected last frame but not this one.  Without this, every
        object the robot ever passed is permanently blocked.
      • BUG 6 (OVER-SENSITIVE PIVOT): The boolean `_is_pivoting` and its
        dual thresholds (theta_high / theta_low) are now properties of the
        planner so the controller can query them cleanly.
    """

    def __init__(self, x_bounds=(-10.0, 12.0), y_bounds=(-5.5, 5.0), resolution=0.15):
        self.res   = resolution
        self.x_min, self.x_max = x_bounds
        self.y_min, self.y_max = y_bounds
        self.g, self.rhs, self.U = {}, {}, {}
        self.km = 0.0
        self.obstacles: set = set()
        self._prev_frame_obstacles: set = set()   # NEW – for symmetric clearing
        self.path_waypoints = []
        self.start_cell = None
        self.goal_cell  = None

    # ── grid ↔ world conversions ──────────────────────────────────────────────
    def w_to_g(self, coord):
        return (
            int(round((coord[0] - self.x_min) / self.res)),
            int(round((coord[1] - self.y_min) / self.res)),
        )

    def g_to_w(self, cell):
        return [self.x_min + cell[0] * self.res, self.y_min + cell[1] * self.res]

    def get_neighbors(self, u):
        neighbors = []
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            v = (u[0]+dx, u[1]+dy)
            wx, wy = self.g_to_w(v)
            if self.x_min <= wx <= self.x_max and self.y_min <= wy <= self.y_max:
                neighbors.append(v)
        return neighbors

    def h(self, s1, s2):
        return math.hypot(s1[0]-s2[0], s1[1]-s2[1])

    def calculate_key(self, s):
        g_rhs = min(self.g.get(s, float('inf')), self.rhs.get(s, float('inf')))
        return (g_rhs + self.h(self.start_cell, s) + self.km, g_rhs)

    def initialize(self, start_w, goal_w):
        self.g.clear(); self.rhs.clear(); self.U.clear()
        self.km = 0.0
        self.obstacles.clear()
        self._prev_frame_obstacles.clear()   # reset clearing buffer on lane start
        self.path_waypoints = []
        self.start_cell = self.w_to_g(start_w)
        self.goal_cell  = self.w_to_g(goal_w)
        self.rhs[self.goal_cell] = 0.0
        self.U[self.goal_cell]   = self.calculate_key(self.goal_cell)

    def cost(self, u, v):
        if u in self.obstacles or v in self.obstacles:
            return float('inf')
        return self.h(u, v)

    def update_vertex(self, u):
        if u != self.goal_cell:
            neighbors = self.get_neighbors(u)
            if neighbors:
                self.rhs[u] = min(
                    self.g.get(v, float('inf')) + self.cost(u, v)
                    for v in neighbors
                )
            else:
                self.rhs[u] = float('inf')
        if u in self.U:
            del self.U[u]
        if self.g.get(u, float('inf')) != self.rhs.get(u, float('inf')):
            self.U[u] = self.calculate_key(u)

    def compute_shortest_path(self):
        iterations = 0
        while self.U:
            top_key = min(self.U.values())
            start_key = self.calculate_key(self.start_cell)
            if top_key >= start_key and \
               self.rhs.get(self.start_cell, float('inf')) == \
               self.g.get(self.start_cell, float('inf')):
                break
            u     = min(self.U, key=self.U.get)
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
            iterations += 1
            if iterations > 50000:   # guard against infinite loop on unsolvable maps
                break

    def generate_waypoints(self):
        path = []
        curr = self.start_cell
        visited = set()
        while curr != self.goal_cell and len(path) < 400:
            if curr in visited:
                break
            visited.add(curr)
            path.append(self.g_to_w(curr))
            neighbors = self.get_neighbors(curr)
            if not neighbors:
                break
            next_node = min(
                neighbors,
                key=lambda v: self.cost(curr, v) + self.g.get(v, float('inf'))
            )
            if (self.cost(curr, next_node) + self.g.get(next_node, float('inf'))) == float('inf'):
                break
            curr = next_node
        path.append(self.g_to_w(self.goal_cell))
        self.path_waypoints = path

    def sense_and_replan(self, current_w, prox_readings, current_yaw):
        """
        Update the occupancy map from the 8 proximity sensor readings and
        replan if anything changed.

        FIX – BUG 5 (STALE MAP):
        We compute the set of cells detected THIS frame and compare it
        against what we stored last frame.  Cells that disappeared are
        removed from the map (and their graph edges updated), so temporary
        obstacles – like the moving walls – are correctly erased once the
        robot drives past them.
        """
        self.start_cell = self.w_to_g(current_w)
        current_frame_cells: set = set()

        # Threshold: ps reading > 150 means an object is within ~0.096 m.
        OBSTACLE_THRESHOLD = 150.0

        for idx, reading in enumerate(prox_readings):
            if reading <= OBSTACLE_THRESHOLD:
                continue
            dist       = ps_to_dist(reading)
            global_ang = current_yaw + PS_ANGLES[idx]
            ox = current_w[0] + dist * math.cos(global_ang)
            oy = current_w[1] + dist * math.sin(global_ang)
            obs_cell = self.w_to_g([ox, oy])

            # 3×3 inflation
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    cell = (obs_cell[0]+dx, obs_cell[1]+dy)
                    if cell != self.start_cell and cell != self.goal_cell:
                        current_frame_cells.add(cell)

        # --- NEW cells detected this frame ---
        added   = current_frame_cells - self.obstacles
        # --- cells that were present last frame but not now (cleared) ---
        cleared = self._prev_frame_obstacles - current_frame_cells

        changed = False
        for cell in added:
            self.obstacles.add(cell)
            self.update_vertex(cell)
            for n in self.get_neighbors(cell):
                self.update_vertex(n)
            changed = True

        for cell in cleared:
            self.obstacles.discard(cell)
            self.update_vertex(cell)
            for n in self.get_neighbors(cell):
                self.update_vertex(n)
            changed = True

        self._prev_frame_obstacles = current_frame_cells

        if changed:
            print("🧱 D* LITE MAP UPDATE: Recalculating …")
            self.compute_shortest_path()
            self.generate_waypoints()


dstar = DStarLitePlanner()

# Hysteresis state for D* Lite pivoting (BUG 6 fix lives here + in the loop).
_is_pivoting = False

# ── 9. LANE RESET ─────────────────────────────────────────────────────────────
def reset_to_lane(lane_idx):
    global lane_start_sim_time, lane_distance_traveled, previous_position
    global last_stall_check_time, last_stall_check_position, _is_pivoting

    start_pos = START_ARRAY[lane_idx]
    goal_pos  = GOAL_ARRAY[lane_idx]
    print(f"🚀 TELEPORTING ROBOT TO LANE {lane_idx} → {start_pos}")
    robot_node.getField("translation").setSFVec3f(start_pos)
    robot_node.getField("rotation").setSFRotation(START_ROT)
    robot_node.resetPhysics()
    for w in wheels:
        w.setVelocity(0.0)

    _is_pivoting = False   # clear hysteresis state on lane reset

    if robot.getTime() > 0.5 and NAVIGATION_MODE == "PURE_DSTAR":
        dstar.initialize(start_pos[:2], goal_pos[:2])
        dstar.compute_shortest_path()
        dstar.generate_waypoints()

    lane_start_sim_time    = robot.getTime()
    lane_distance_traveled = 0.0
    previous_position      = np.array([start_pos[0], start_pos[1]])
    last_stall_check_time  = robot.getTime()
    last_stall_check_position = np.array([start_pos[0], start_pos[1]])


# ── 10. STALL / TIMEOUT PARAMETERS ───────────────────────────────────────────
STALL_DISTANCE_THRESHOLD = 0.15   # must travel at least this far every check
STALL_CHECK_INTERVAL     = 10.0   # seconds between stall checks

# ── 11. STARTUP ──────────────────────────────────────────────────────────────
reset_to_lane(current_lane_index)

# Let physics settle for 1 simulated second before starting.
t0 = robot.getTime()
while robot.step(timestep) != -1:
    if robot.getTime() - t0 >= 1.0:
        break

# APF tuning parameters
K_ATTRACTIVE  = 2.5
K_REPULSIVE   = 0.08
INFLUENCE_DIST = 0.45   # metres – prox sensor range is 0.10 m, but use wider for early braking

last_metric_print_time = 0.0

print(f"🚀 MIST Controller active | mode: {NAVIGATION_MODE}")

# ── 12. MAIN NAVIGATION LOOP ──────────────────────────────────────────────────
try:
    while robot.step(timestep) != -1:
        pos              = robot_node.getPosition()
        curr_time        = robot.getTime()
        current_pos_2d   = np.array([pos[0], pos[1]])

        # Odometry
        if previous_position is not None:
            d = np.linalg.norm(current_pos_2d - previous_position)
            if d < 0.5:
                lane_distance_traveled += d
        previous_position = current_pos_2d

        time_in_lane = curr_time - lane_start_sim_time

        if camera:
            camera.getImage()

        # ── Helper: log + advance (or replay) lane ──
        def advance_or_hold_lane(log_status):
            global current_lane_index
            csv_writer.writerow([
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                current_lane_index, log_status,
                round(time_in_lane, 3), round(lane_distance_traveled, 3)
            ])
            csv_file.flush()
            if DEBUG_HOLD_LANE:
                print(f"🔄 DEBUG RETRY: Re-running Lane {current_lane_index}.")
            else:
                current_lane_index = (current_lane_index + 1) % num_lanes
            reset_to_lane(current_lane_index)

        # ── Stall detection ──
        if curr_time - last_stall_check_time >= STALL_CHECK_INTERVAL:
            moved = np.linalg.norm(current_pos_2d - last_stall_check_position)
            if moved < STALL_DISTANCE_THRESHOLD:
                print(f"🛑 STALL: only {moved:.3f} m in {STALL_CHECK_INTERVAL}s.")
                advance_or_hold_lane(f"{NAVIGATION_MODE}_LOCAL_MINIMA_STALL")
                if current_lane_index == 0 and not DEBUG_HOLD_LANE:
                    break
                continue
            last_stall_check_time     = curr_time
            last_stall_check_position = current_pos_2d

        # ── Goal check ──
        if pos[0] >= GOAL_ARRAY[current_lane_index][0]:
            print(f"🏁 GOAL: Lane {current_lane_index} done in {time_in_lane:.2f}s")
            csv_writer.writerow([
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                current_lane_index, NAVIGATION_MODE,
                round(time_in_lane, 3), round(lane_distance_traveled, 3)
            ])
            csv_file.flush()
            current_lane_index = (current_lane_index + 1) % num_lanes
            if current_lane_index == 0:
                print("🎯 All lanes complete.")
                break
            reset_to_lane(current_lane_index)
            continue

        # ── Sensor readings ──────────────────────────────────────────────────
        #
        # FIX – BUG 2 (WRONG YAW):
        # getOrientation() returns a flat row-major 3×3 rotation matrix R.
        # Index layout: [r00 r01 r02  r10 r11 r12  r20 r21 r22]
        #                  0   1   2    3   4   5    6   7   8
        # Yaw (rotation around world-Z, CCW positive) = atan2(r10, r00)
        #                                             = atan2(R[3], R[0])
        #
        # The original code subtracted pi/2 from this result, which introduced
        # a constant 90° offset: a robot facing +X would report yaw = -90°
        # instead of 0°, making all downstream heading errors wrong.
        # The fix is simply to remove that subtraction.
        #
        rot_matrix  = robot_node.getOrientation()
        current_yaw = math.atan2(rot_matrix[3], rot_matrix[0])
        # Normalise to (-π, π] – already guaranteed by atan2, but explicit wrap
        # is kept for clarity.
        current_yaw = (current_yaw + math.pi) % (2.0 * math.pi) - math.pi

        prox_readings = np.array([ps.getValue() for ps in proximity_sensors])

        # ── ALGORITHM 1: ARTIFICIAL POTENTIAL FIELDS ─────────────────────────
        if NAVIGATION_MODE == "PURE_APF":
            # ── Attractive force (goal) ──────────────────────────────────────
            target_pos  = np.array([GOAL_ARRAY[current_lane_index][0],
                                    GOAL_ARRAY[current_lane_index][1]])
            vec_to_goal = target_pos - current_pos_2d
            dist_to_goal = np.linalg.norm(vec_to_goal)
            if dist_to_goal > 0:
                f_att = K_ATTRACTIVE * (vec_to_goal / dist_to_goal)
            else:
                f_att = np.array([0.0, 0.0])

            # ── Repulsive forces (obstacles via ps sensors) ──────────────────
            #
            # FIX note (APF sensing): The original code iterated over LiDAR
            # rays which don't exist on the e-puck.  We iterate the 8 ps
            # sensors instead.  Each sensor contributes a repulsive vector
            # pointing directly away from the detected obstacle, scaled by the
            # standard potential-field formula: K * (1/d - 1/d0) / d².
            #
            f_rep = np.array([0.0, 0.0])
            for idx, reading in enumerate(prox_readings):
                dist = ps_to_dist(reading)
                if dist >= INFLUENCE_DIST:
                    continue  # obstacle outside influence zone – ignore
                # Direction from robot toward obstacle (world frame)
                global_ang  = current_yaw + PS_ANGLES[idx]
                vec_toward  = np.array([math.cos(global_ang), math.sin(global_ang)])
                vec_away    = -vec_toward

                # Standard gradient of repulsive potential:  K*(1/d - 1/d0)/d²
                factor = (1.0 / dist) - (1.0 / INFLUENCE_DIST)
                mag    = K_REPULSIVE * factor / (dist ** 2)
                f_rep += mag * vec_away

            f_total = f_att + f_rep

            # ── Heading error & speed ────────────────────────────────────────
            desired_heading  = math.atan2(f_total[1], f_total[0])
            heading_error    = (desired_heading - current_yaw + math.pi) % (2.0 * math.pi) - math.pi

            force_mag = np.linalg.norm(f_total)

            # FIX – BUG 4 (APF STALL):
            # The original code had a hard minimum velocity of 1.0 rad/s.
            # This meant the robot was always pushed forward even when the
            # heading error was large (e.g. 170°), causing it to ram walls
            # while trying to turn.
            #
            # Fix: velocity is proportional to force magnitude with NO hard
            # floor.  Additionally, when the heading error exceeds 60°, we
            # scale down even further so the robot can pivot cleanly.
            base_velocity = np.clip(force_mag * 1.2, 0.0, 4.0)
            if abs(heading_error) > math.radians(60):
                base_velocity *= 0.15   # allow near-stop for sharp turns
            elif abs(heading_error) > math.radians(30):
                base_velocity *= 0.50   # moderate slowdown for medium turns

            angular_velocity = np.clip(heading_error * 3.5, -2.5, 2.5)

        # ── ALGORITHM 2: D* LITE ──────────────────────────────────────────────
        elif NAVIGATION_MODE == "PURE_DSTAR":
            # Update map and replan (BUG 5 fix lives inside sense_and_replan).
            dstar.sense_and_replan(pos[:2], prox_readings, current_yaw)

            # Waypoint tracking
            if dstar.path_waypoints:
                active_wp     = dstar.path_waypoints[0]
                dist_to_wp    = math.hypot(active_wp[0]-pos[0], active_wp[1]-pos[1])
                if dist_to_wp < 0.26 and len(dstar.path_waypoints) > 1:
                    dstar.path_waypoints.pop(0)
                    active_wp  = dstar.path_waypoints[0]
                    dist_to_wp = math.hypot(active_wp[0]-pos[0], active_wp[1]-pos[1])
            else:
                active_wp  = [GOAL_ARRAY[current_lane_index][0],
                               GOAL_ARRAY[current_lane_index][1]]
                dist_to_wp = math.hypot(active_wp[0]-pos[0], active_wp[1]-pos[1])

            desired_heading = math.atan2(active_wp[1]-pos[1], active_wp[0]-pos[0])
            heading_error   = (desired_heading - current_yaw + math.pi) % (2.0*math.pi) - math.pi

            # Front proximity: use ps0 (right-front) and ps7 (left-front).
            front_dist = min(ps_to_dist(prox_readings[0]), ps_to_dist(prox_readings[7]))

            # FIX – BUG 6 (OVER-SENSITIVE PIVOT / HYSTERESIS):
            #
            # Original: pivot (base=0) whenever |heading_error| > 0.20 rad (11°).
            # At resolution 0.15 m the planner regularly produces waypoints that
            # require a 10-20° correction, so the robot was almost always stopped.
            #
            # Fix: dual-threshold hysteresis state machine (as described in the
            # paper):
            #   - Engage pivot when |error| rises above THETA_HIGH (≈ 30°)
            #   - Release pivot only when |error| drops below THETA_LOW  (≈  8°)
            #   - While not pivoting, forward speed is NOT zeroed – just scaled.
            #
            THETA_HIGH = math.radians(30)   # 0.52 rad – engage in-place rotation
            THETA_LOW  = math.radians( 8)   # 0.14 rad – release pivot

            # --- FIXED PIVOT / HYSTERESIS STATE MACHINE ---
            THETA_HIGH = math.radians(30)  # 0.52 rad – engage in-place rotation
            THETA_LOW = math.radians(8)  # 0.14 rad – release pivot

            if abs(heading_error) > THETA_HIGH:
                _is_pivoting = True
            elif _is_pivoting and abs(heading_error) < THETA_LOW:
                _is_pivoting = False
            # If in the dead-band (THETA_LOW <= |err| <= THETA_HIGH), keep previous state.

            if _is_pivoting:
                base_velocity = 0.0   # pure in-place rotation
            else:
                base_velocity = 3.0
                # Slow down near waypoint
                if dist_to_wp < 0.40:
                    base_velocity = max(0.5, base_velocity * (dist_to_wp / 0.40))
                # Slow down near front obstacle
                if front_dist < 0.40:
                    base_velocity *= (front_dist / 0.40)

            angular_velocity = np.clip(heading_error * 7.5, -4.0, 4.0)

        else:
            # Unknown mode – coast to stop
            base_velocity    = 0.0
            angular_velocity = 0.0

        # ── WHEEL MIXING ──────────────────────────────────────────────────────
        #
        # FIX – BUG 3 (INVERTED MIXER):
        # The original code had:
        #   wheels[0] = base - angular    (left wheel SLOWER when angular > 0)
        #   wheels[1] = base + angular    (right wheel FASTER when angular > 0)
        # That combination turns the robot CLOCKWISE (right) when angular > 0,
        # but a positive heading_error means the robot needs to turn LEFT
        # (counter-clockwise).  The two were therefore always fighting each other.
        #
        # Correct convention for a standard differential drive (CCW yaw positive):
        #   left_wheel  = base + angular   → faster left  = CCW = positive yaw
        #   right_wheel = base - angular   → slower right = CCW
        #
        wheels[0].setVelocity(np.clip(base_velocity + angular_velocity, -MAX_SPEED, MAX_SPEED))
        wheels[1].setVelocity(np.clip(base_velocity - angular_velocity, -MAX_SPEED, MAX_SPEED))

except Exception as e:
    print(f"❌ CRITICAL RUNTIME EXCEPTION: {e}")
    raise e

finally:
    print("💾 Closing performance data streams.")
    csv_file.close()
    robot.simulationQuit(0)