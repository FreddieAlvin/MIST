"""
mist_controller.py – MIST navigation controller.

Supports three MIST_NAV_MODE values (set via environment variable):
    PURE_DSTAR   – D* Lite incremental path planner
    PURE_APF     – Artificial Potential Fields
    PPO          – Trained Stable-Baselines3 MultiInputPolicy agent

Logging (CSV columns):
    Timestamp_Wall_Clock | Lane_Index | Navigation_Mode |
    Time_Taken_Seconds   | Distance_Traveled_Meters | Lane_Completed

Lane_Completed = True  if the robot reached GOAL_X before the time/stall limit.
Lane_Completed = False if the lane was abandoned due to stall or timeout.
"""

import os
import sys
import math
import csv
import datetime
from pathlib import Path

# ── 1. Path injection (must be first) ────────────────────────────────────────
CONTROLLER_DIR = Path(__file__).resolve().parent
SRC_DIR        = CONTROLLER_DIR.parent.parent   # MIST/src/
MIST_ROOT      = SRC_DIR.parent                 # MIST/

# mist_env.py lives in MIST/src/utils/ — add it explicitly (train_ppo.py does
# the same). Without this, `from mist_env import ...` fails in PPO mode.
_path_candidates = [SRC_DIR / "utils", SRC_DIR, MIST_ROOT, CONTROLLER_DIR]

# Defensive fallback: walk up a few levels and grab any dir that actually
# contains mist_env.py, so the import works regardless of exact tree layout.
for _up in [CONTROLLER_DIR, *list(CONTROLLER_DIR.parents)[:6]]:
    for _cand in (_up, _up / "utils"):
        if (_cand / "mist_env.py").exists():
            _path_candidates.insert(0, _cand)

for path in _path_candidates:
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

# ── 2. Webots platform path ───────────────────────────────────────────────────
if sys.platform == "darwin":
    os.environ['WEBOTS_HOME'] = '/Applications/Webots.app'
    _wlib = os.path.join(os.environ['WEBOTS_HOME'],
                         'Contents', 'lib', 'controller', 'python')
elif sys.platform == "win32":
    os.environ['WEBOTS_HOME'] = 'C:\\Program Files\\Webots'
    _wlib = os.path.join(os.environ['WEBOTS_HOME'],
                         'lib', 'controller', 'python')
else:
    os.environ['WEBOTS_HOME'] = '/usr/local/webots'
    _wlib = os.path.join(os.environ['WEBOTS_HOME'],
                         'lib', 'controller', 'python')

if os.path.exists(_wlib) and _wlib not in sys.path:
    sys.path.append(_wlib)

from controller import Supervisor
import numpy as np

# ── 3. Testing control toggles ────────────────────────────────────────────────
DEBUG_HOLD_LANE = False   # True → repeat same lane indefinitely (dev only)

# ── 4. Logging & file management ─────────────────────────────────────────────
LOG_DIR   = MIST_ROOT / "results" / "logs"
MODEL_DIR = MIST_ROOT / "saved_models"
LOG_DIR.mkdir(parents=True, exist_ok=True)

NAVIGATION_MODE = os.environ.get("MIST_NAV_MODE", "PURE_DSTAR")
CSV_PATH        = LOG_DIR / f"{NAVIGATION_MODE}_performance_log.csv"
file_exists     = CSV_PATH.exists()

csv_file   = open(str(CSV_PATH), mode='a', newline='', encoding='utf-8')
csv_writer = csv.writer(csv_file)
if not file_exists:
    csv_writer.writerow([
        "Timestamp_Wall_Clock", "Lane_Index", "Navigation_Mode",
        "Time_Taken_Seconds",   "Distance_Traveled_Meters", "Lane_Completed",
        "Obstacle_Hits"
    ])
    csv_file.flush()

# ── 5. Core initialisation ────────────────────────────────────────────────────
robot      = Supervisor()
timestep   = int(robot.getBasicTimeStep())
robot_node = robot.getSelf()

if robot_node is None:
    print("❌ SYSTEM CRITICAL: robot.getSelf() returned None! Grant supervisor privileges.")
    sys.exit(1)

# ── 6. Hardware discovery ─────────────────────────────────────────────────────
wheels = [None, None]
discovered_motors = []
for i in range(robot.getNumberOfDevices()):
    device = robot.getDeviceByIndex(i)
    if device.__class__.__name__ == "Motor":
        discovered_motors.append(device)

for motor in discovered_motors:
    name = motor.getName().lower()
    if "left" in name or "m1" in name:
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

# 8 proximity sensors
proximity_sensors = []
for i in range(8):
    ps = robot.getDevice(f"ps{i}")
    if ps is None:
        print(f"❌ SYSTEM CRITICAL: Proximity sensor ps{i} not found!")
        sys.exit(1)
    ps.enable(timestep)
    proximity_sensors.append(ps)

# Sensor angles in robot frame (e-puck spec, radians CCW-positive)
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
PS_MAX_RANGE = 0.10  # metres

def ps_to_dist(reading: float) -> float:
    """Raw ADC [0..4096] → estimated distance in metres."""
    return max(0.01, PS_MAX_RANGE * (1.0 - min(reading / 4096.0, 1.0)))

# Lidar (360-ray, 3 m) – the PPO policy's main obstacle sensor.
lidar = robot.getDevice("lidar")
if lidar:
    lidar.enable(timestep)
    # Point cloud gives obstacle positions directly in the sensor frame
    # (convention-free), used by APF and D* for ranged obstacle sensing.
    try:
        lidar.enablePointCloud()
    except Exception:
        pass
    print(f"✅ Lidar enabled ({lidar.getHorizontalResolution()} rays, "
          f"{lidar.getMaxRange():.1f} m range)")
elif NAVIGATION_MODE in ("PPO", "PURE_APF", "PURE_DSTAR"):
    print("⚠️  No Lidar found – falling back to short-range proximity sensors.")


def lidar_obstacle_points(max_range: float):
    """
    Return obstacle points (x_forward, y_left) in the ROBOT frame from the Lidar
    point cloud, filtered to 0.06 m < range < max_range. Convention-free: each
    point is a real position relative to the sensor, so no angle bookkeeping.
    Returns [] if the Lidar is unavailable.
    """
    if lidar is None:
        return []
    pts = []
    try:
        cloud = lidar.getPointCloud()
    except Exception:
        return []
    for p in cloud:
        d = math.hypot(p.x, p.y)
        if 0.06 < d < max_range:
            pts.append((p.x, p.y, d))
    return pts

# Camera (optional for APF/D* modes; required for PPO only if USE_CAMERA)
camera = robot.getDevice("camera")
if camera:
    camera.enable(timestep)
    print(f"✅ Camera enabled ({camera.getWidth()}×{camera.getHeight()} px)")
else:
    print("⚠️  No camera found.")
    if NAVIGATION_MODE == "PPO":
        print("   PPO will use zero-image tensors (camera observation = 0).")

# ── 7. Environment layout (4 lanes) ──────────────────────────────────────────
START_ARRAY = [
    [-8.8541,  3.13013,  -0.000217358],   # Lane 0
    [-8.8541,  0.370129, -0.000217358],   # Lane 1
    [-8.8541, -2.36987,  -0.000217358],   # Lane 2
    [-8.8541, -5.12987,  -0.000217358],   # Lane 3
]
GOAL_ARRAY = [
    [10.6659,  3.13013,  -0.000217358],
    [10.6659,  0.370129, -0.000217358],
    [10.6659, -2.36987,  -0.000217358],
    [10.6659, -5.12987,  -0.000217358],
]
START_ROT = [0, 0, 1, 0]

num_lanes            = len(START_ARRAY)
current_lane_index   = 0
MAX_SPEED            = 6.28      # e-puck motor limit (rad/s)
lane_start_sim_time  = 0.0
lane_distance_traveled = 0.0
previous_position    = None

# ── Obstacle-hit tracking ─────────────────────────────────────────────────────
# A "hit" is counted once per distinct contact (rising edge), not every step, so
# grinding along a wall for 1 s doesn't inflate the count. COLLISION_RAW mirrors
# the env's collision threshold (normalised 0.45 × PROX_MAX ≈ 1843 ADC).
COLLISION_RAW    = 0.45 * 4096.0
lane_hit_count   = 0
prev_in_collision = False

last_stall_check_time     = 0.0
last_stall_check_position = np.array([0.0, 0.0])

# ── 8. PPO model loading (only for PPO mode) ──────────────────────────────────
ppo_model = None
if NAVIGATION_MODE == "PPO":
    try:
        from stable_baselines3 import PPO as SB3PPO
        # Pull the SAME action-scaling constants the policy was trained with,
        # so eval-time wheel mapping cannot drift from training.
        from mist_env import (MistNavEnv, IMAGE_H, IMAGE_W,
                              FORWARD_SCALE, TURN_SCALE, USE_CAMERA, LIDAR_SECTORS)
        model_path = MODEL_DIR / "ppo_mist_optimal_model"
        ppo_model = SB3PPO.load(str(model_path))

        # Guard: the saved policy's observation space must match the current
        # config (USE_CAMERA + lidar sector count). A mismatch otherwise throws
        # a cryptic error deep inside SB3 mid-episode. Fail early instead.
        try:
            _spaces = ppo_model.observation_space.spaces
            _model_keys  = set(_spaces.keys())
            _model_lidar = int(_spaces["lidar"].shape[0]) if "lidar" in _spaces else None
        except AttributeError:
            _model_keys, _model_lidar = set(), None
        _model_wants_image = "image" in _model_keys
        _mismatch = (_model_wants_image != USE_CAMERA) or \
                    (_model_lidar is not None and _model_lidar != LIDAR_SECTORS)
        if _mismatch:
            print("❌ Model / config mismatch — the saved model was trained with "
                  "different settings:")
            print(f"   image obs: model={_model_wants_image}  vs  USE_CAMERA={USE_CAMERA}")
            print(f"   lidar size: model={_model_lidar}  vs  LIDAR_SECTORS={LIDAR_SECTORS}")
            print("   → Retrain so the model matches the current settings:")
            print("        python <…>/src/agents/train_ppo.py")
            sys.exit(1)

        print(f"✅ PPO model loaded from: {model_path}.zip  "
              f"(image obs: {_model_wants_image}, lidar sectors: {_model_lidar})")
    except Exception as e:
        print(f"❌ Failed to load PPO model: {e}")
        sys.exit(1)

def _build_ppo_obs(pos, prox_readings, current_yaw, lane_idx):
    """
    Build the Dict observation that matches MistNavEnv's observation_space.
    Mirrors _get_obs() from mist_env.py exactly.
    """
    from mist_env import (GOAL_X, LANE_STARTS, MAX_WHEEL_SPEED, GOAL_DIST_NORM,
                          LIDAR_SECTORS, LIDAR_MAX_RANGE)

    # ── lidar sectors (mirror MistNavEnv._get_lidar_sectors exactly) ──
    if lidar is not None:
        ranges = np.asarray(lidar.getRangeImage(), dtype=np.float32)
        if ranges.size:
            ranges  = np.where(np.isfinite(ranges), ranges, LIDAR_MAX_RANGE)
            ranges  = np.clip(ranges, 0.0, LIDAR_MAX_RANGE)
            sectors = np.array([float(np.min(g))
                                for g in np.array_split(ranges, LIDAR_SECTORS)],
                               dtype=np.float32)
            lidar_obs = (sectors / LIDAR_MAX_RANGE).astype(np.float32)
        else:
            lidar_obs = np.ones(LIDAR_SECTORS, dtype=np.float32)
    else:
        lidar_obs = np.ones(LIDAR_SECTORS, dtype=np.float32)

    # ── image (only when the trained policy expects it) ──
    if USE_CAMERA and camera is not None:
        raw = camera.getImage()
        w   = camera.getWidth()
        h   = camera.getHeight()
        if raw and len(raw) > 0:
            arr   = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 4))
            rgb   = arr[:, :, 2::-1]
            ri    = (np.arange(IMAGE_H) * h / IMAGE_H).astype(int)
            ci    = (np.arange(IMAGE_W) * w / IMAGE_W).astype(int)
            frame = rgb[np.ix_(ri, ci)]
            # uint8 [0,255] – must match MistNavEnv's image space exactly.
            image = np.ascontiguousarray(frame, dtype=np.uint8)
        else:
            image = np.zeros((IMAGE_H, IMAGE_W, 3), dtype=np.uint8)
    else:
        image = None

    # ── vector ──
    v_l = wheels[0].getVelocity()
    v_r = wheels[1].getVelocity()
    v_mean = np.clip((v_l + v_r) / (2.0 * MAX_WHEEL_SPEED), -1.0, 1.0)
    v_diff = np.clip((v_l - v_r) / MAX_WHEEL_SPEED,          -1.0, 1.0)

    goal_x = GOAL_X
    goal_y = LANE_STARTS[lane_idx][1]

    target_angle  = math.atan2(goal_y - pos[1], goal_x - pos[0])
    heading_error = (target_angle - current_yaw + math.pi) % (2 * math.pi) - math.pi
    heading_norm  = float(np.clip(heading_error / math.pi, -1.0, 1.0))

    rel_x = float(np.clip((goal_x - pos[0]) / 20.0, -1.0, 1.0))
    rel_y = float(np.clip((goal_y - pos[1]) / 20.0, -1.0, 1.0))
    dist  = math.hypot(goal_x - pos[0], goal_y - pos[1])
    dist_norm = float(np.clip(dist / GOAL_DIST_NORM, 0.0, 1.0))

    vector = np.array([v_mean, v_diff, heading_norm, rel_x, rel_y, dist_norm],
                      dtype=np.float32)

    obs = {"lidar": lidar_obs, "vector": vector}
    if USE_CAMERA:
        obs["image"] = image
    return obs

# ── 9. D* Lite planner ────────────────────────────────────────────────────────
class DStarLitePlanner:
    """
    Incremental D* Lite on a 2-D occupancy grid.

    Fixes vs original:
      • BUG 5 (STALE MAP): sense_and_replan() maintains _prev_frame_obstacles
        and symmetrically REMOVES cells no longer detected this frame.
      • BUG 6 (OVER-SENSITIVE PIVOT): hysteresis thresholds moved to planner
        so the controller can query _is_pivoting cleanly.
    """

    def __init__(self, x_bounds=(-10.0, 12.0), y_bounds=(-6.5, 5.5), resolution=0.15):
        self.res   = resolution
        self.x_min, self.x_max = x_bounds
        self.y_min, self.y_max = y_bounds
        self.g, self.rhs, self.U = {}, {}, {}
        self.km = 0.0
        self.obstacles: set            = set()
        self._prev_frame_obstacles: set = set()
        self.path_waypoints = []
        self.start_cell = None
        self.goal_cell  = None
        self.last_start = None   # for km accumulation as the robot moves

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
        self._prev_frame_obstacles.clear()
        self.path_waypoints = []
        self.start_cell = self.w_to_g(start_w)
        self.goal_cell  = self.w_to_g(goal_w)
        self.last_start = self.start_cell
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
            top_key   = min(self.U.values())
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
            if iterations > 50_000:
                break

    def generate_waypoints(self):
        path    = []
        curr    = self.start_cell
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

    def sense_and_replan(self, current_w, prox_readings, current_yaw,
                         lidar_points=None):
        # D* Lite: when the start (robot) moves, accumulate km so priority keys
        # stay consistent with the heuristic measured from the new start.
        new_start = self.w_to_g(current_w)
        if self.last_start is not None:
            self.km += self.h(self.last_start, new_start)
        self.last_start = new_start
        self.start_cell = new_start
        current_frame_cells: set = set()

        cy, sy = math.cos(current_yaw), math.sin(current_yaw)

        if lidar_points:
            # Ranged sensing (≤ ~2.5 m): map every Lidar hit into the grid, so the
            # planner can route around obstacles it can SEE rather than ones it has
            # already bumped. This is the key D* upgrade over 10 cm proximity.
            for px, py, _d in lidar_points:
                ox = current_w[0] + (cy * px - sy * py)
                oy = current_w[1] + (sy * px + cy * py)
                obs_cell = self.w_to_g([ox, oy])
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        cell = (obs_cell[0] + dx, obs_cell[1] + dy)
                        if cell != self.start_cell and cell != self.goal_cell:
                            current_frame_cells.add(cell)
        else:
            OBSTACLE_THRESHOLD = 150.0
            for idx, reading in enumerate(prox_readings):
                if reading <= OBSTACLE_THRESHOLD:
                    continue
                dist       = ps_to_dist(reading)
                global_ang = current_yaw + PS_ANGLES[idx]
                ox = current_w[0] + dist * math.cos(global_ang)
                oy = current_w[1] + dist * math.sin(global_ang)
                obs_cell = self.w_to_g([ox, oy])
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        cell = (obs_cell[0]+dx, obs_cell[1]+dy)
                        if cell != self.start_cell and cell != self.goal_cell:
                            current_frame_cells.add(cell)

        added   = current_frame_cells - self.obstacles
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
            self.compute_shortest_path()
            self.generate_waypoints()


dstar        = DStarLitePlanner()
_is_pivoting = False

# ── 10. Lane reset ────────────────────────────────────────────────────────────
def reset_to_lane(lane_idx):
    global lane_start_sim_time, lane_distance_traveled, previous_position
    global last_stall_check_time, last_stall_check_position, _is_pivoting
    global lane_hit_count, prev_in_collision

    start_pos = START_ARRAY[lane_idx]
    goal_pos  = GOAL_ARRAY[lane_idx]
    print(f"🚀 TELEPORTING → Lane {lane_idx} | start={start_pos[:2]}")
    robot_node.getField("translation").setSFVec3f(start_pos)
    robot_node.getField("rotation").setSFRotation(START_ROT)
    robot_node.resetPhysics()
    for w in wheels:
        w.setVelocity(0.0)

    _is_pivoting = False

    if robot.getTime() > 0.5 and NAVIGATION_MODE == "PURE_DSTAR":
        dstar.initialize(start_pos[:2], goal_pos[:2])
        dstar.compute_shortest_path()
        dstar.generate_waypoints()

    lane_start_sim_time        = robot.getTime()
    lane_distance_traveled     = 0.0
    previous_position          = np.array([start_pos[0], start_pos[1]])
    last_stall_check_time      = robot.getTime()
    last_stall_check_position  = np.array([start_pos[0], start_pos[1]])
    lane_hit_count             = 0
    prev_in_collision          = False

# ── 11. Stall / timeout parameters ───────────────────────────────────────────
STALL_DISTANCE_THRESHOLD = 0.15
STALL_CHECK_INTERVAL     = 10.0   # seconds

# ── 12. APF tuning ────────────────────────────────────────────────────────────
# --- APF Lidar upgrade constants ---
# Lidar gives 3 m of lookahead vs the 10 cm proximity sensors, so APF can build a
# smooth repulsive field that pushes the robot away from obstacles early instead
# of reacting at contact. A tangential ("vortex") term breaks the classic APF
# local minimum where attractive and repulsive forces cancel head-on.
INFLUENCE_DIST_LIDAR = 0.9     # m  – repulsion onset distance (Lidar)
K_REPULSIVE_LIDAR    = 0.06    # repulsion gain (Lidar)
K_VORTEX             = 0.5     # tangential gain for local-minimum escape

K_ATTRACTIVE   = 2.5
K_REPULSIVE    = 0.08
INFLUENCE_DIST = 0.45

# ── 13. Startup ───────────────────────────────────────────────────────────────
reset_to_lane(current_lane_index)

# Let physics settle for 1 simulated second
t0 = robot.getTime()
while robot.step(timestep) != -1:
    if robot.getTime() - t0 >= 1.0:
        break

print(f"🚀 MIST Controller active | mode: {NAVIGATION_MODE} | lanes: {num_lanes}")

# ── 14. Main navigation loop ──────────────────────────────────────────────────
try:
    while robot.step(timestep) != -1:
        pos            = robot_node.getPosition()
        curr_time      = robot.getTime()
        current_pos_2d = np.array([pos[0], pos[1]])

        # Odometry
        if previous_position is not None:
            d = np.linalg.norm(current_pos_2d - previous_position)
            if d < 0.5:
                lane_distance_traveled += d
        previous_position = current_pos_2d

        time_in_lane = curr_time - lane_start_sim_time

        # ── Shared helper: log + advance lane ────────────────────────────────
        def log_and_advance(lane_completed: bool):
            """
            Write one CSV row, then move to next lane (or replay if debugging).
            lane_completed=True  → robot reached the goal
            lane_completed=False → stall or timeout
            """
            global current_lane_index
            status = NAVIGATION_MODE if lane_completed else f"{NAVIGATION_MODE}_STALL"
            csv_writer.writerow([
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                current_lane_index,
                status,
                round(time_in_lane, 3),
                round(lane_distance_traveled, 3),
                lane_completed,      # ← Lane_Completed column
                lane_hit_count,      # ← Obstacle_Hits column
            ])
            csv_file.flush()
            if DEBUG_HOLD_LANE:
                print(f"🔄 DEBUG RETRY: Re-running Lane {current_lane_index}.")
            else:
                current_lane_index = (current_lane_index + 1) % num_lanes
            reset_to_lane(current_lane_index)

        # ── Stall detection ───────────────────────────────────────────────────
        if curr_time - last_stall_check_time >= STALL_CHECK_INTERVAL:
            moved = np.linalg.norm(current_pos_2d - last_stall_check_position)
            if moved < STALL_DISTANCE_THRESHOLD:
                print(f"🛑 STALL: only {moved:.3f} m in {STALL_CHECK_INTERVAL}s | "
                      f"Lane {current_lane_index} | hits={lane_hit_count}")
                log_and_advance(lane_completed=False)
                if current_lane_index == 0 and not DEBUG_HOLD_LANE:
                    break
                continue
            last_stall_check_time     = curr_time
            last_stall_check_position = current_pos_2d

        # ── Goal check ───────────────────────────────────────────────────────
        if pos[0] >= GOAL_ARRAY[current_lane_index][0]:
            print(f"🏁 GOAL: Lane {current_lane_index} done in {time_in_lane:.2f}s | "
                  f"{lane_distance_traveled:.2f}m | hits={lane_hit_count}")
            log_and_advance(lane_completed=True)
            if current_lane_index == 0 and not DEBUG_HOLD_LANE:
                print("🎯 All lanes complete.")
                break
            continue

        # ── Sensor readings ───────────────────────────────────────────────────
        # Yaw: atan2(r10, r00) from the 3×3 rotation matrix (flat, row-major)
        rot_matrix  = robot_node.getOrientation()
        current_yaw = math.atan2(rot_matrix[3], rot_matrix[0])
        current_yaw = (current_yaw + math.pi) % (2.0 * math.pi) - math.pi

        prox_readings = np.array([ps.getValue() for ps in proximity_sensors])

        # ── Obstacle-hit counting (rising edge on front sensors ps0,1,6,7) ────
        front_raw    = max(prox_readings[0], prox_readings[1],
                           prox_readings[6], prox_readings[7])
        in_collision = front_raw > COLLISION_RAW
        if in_collision and not prev_in_collision:
            lane_hit_count += 1
        prev_in_collision = in_collision

        # ── ALGORITHM: PURE APF ───────────────────────────────────────────────
        if NAVIGATION_MODE == "PURE_APF":
            target_pos   = np.array([GOAL_ARRAY[current_lane_index][0],
                                     GOAL_ARRAY[current_lane_index][1]])
            vec_to_goal  = target_pos - current_pos_2d
            dist_to_goal = np.linalg.norm(vec_to_goal)
            f_att = K_ATTRACTIVE * (vec_to_goal / dist_to_goal) if dist_to_goal > 0 \
                    else np.array([0.0, 0.0])

            # ── Repulsion from the Lidar point cloud (3 m lookahead) ──────────
            # Each obstacle point pushes the robot away with FIRAS magnitude.
            # Falls back to the 10 cm proximity sensors if no Lidar.
            f_rep = np.array([0.0, 0.0])
            cy, sy = math.cos(current_yaw), math.sin(current_yaw)
            pts = lidar_obstacle_points(INFLUENCE_DIST_LIDAR)
            if pts:
                for px, py, d in pts:
                    # away direction in robot frame, rotated to world frame
                    ax_r, ay_r = -px / d, -py / d
                    ax = cy * ax_r - sy * ay_r
                    ay = sy * ax_r + cy * ay_r
                    factor = (1.0 / d) - (1.0 / INFLUENCE_DIST_LIDAR)
                    mag    = K_REPULSIVE_LIDAR * factor / (d ** 2)
                    f_rep += mag * np.array([ax, ay])
            else:
                for idx, reading in enumerate(prox_readings):
                    dist = ps_to_dist(reading)
                    if dist >= INFLUENCE_DIST:
                        continue
                    global_ang = current_yaw + PS_ANGLES[idx]
                    vec_away   = -np.array([math.cos(global_ang), math.sin(global_ang)])
                    factor = (1.0 / dist) - (1.0 / INFLUENCE_DIST)
                    mag    = K_REPULSIVE * factor / (dist ** 2)
                    f_rep += mag * vec_away

            # ── Vortex term: escape the head-on local minimum ─────────────────
            # When repulsion nearly cancels attraction (obstacle dead ahead), add
            # a tangential force perpendicular to the goal direction so the robot
            # consistently slips around one side instead of stalling.
            rep_mag = np.linalg.norm(f_rep)
            if rep_mag > 1e-6 and dist_to_goal > 1e-6:
                att_hat = f_att / (np.linalg.norm(f_att) + 1e-9)
                rep_hat = f_rep / rep_mag
                if float(np.dot(att_hat, rep_hat)) < -0.6:   # forces oppose
                    # rotate attractive dir +90° → consistent left bias
                    tangent = np.array([-att_hat[1], att_hat[0]])
                    f_rep  += K_VORTEX * rep_mag * tangent

            f_total = f_att + f_rep
            desired_heading = math.atan2(f_total[1], f_total[0])
            heading_error   = (desired_heading - current_yaw + math.pi) % (2.0*math.pi) - math.pi
            force_mag = np.linalg.norm(f_total)

            base_velocity = np.clip(force_mag * 1.2, 0.0, 4.0)
            if abs(heading_error) > math.radians(60):
                base_velocity *= 0.15
            elif abs(heading_error) > math.radians(30):
                base_velocity *= 0.50

            angular_velocity = np.clip(heading_error * 3.5, -2.5, 2.5)

        # ── ALGORITHM: PURE D* LITE ───────────────────────────────────────────
        elif NAVIGATION_MODE == "PURE_DSTAR":
            dstar.sense_and_replan(pos[:2], prox_readings, current_yaw,
                                   lidar_points=lidar_obstacle_points(2.5))

            if dstar.path_waypoints:
                active_wp  = dstar.path_waypoints[0]
                dist_to_wp = math.hypot(active_wp[0]-pos[0], active_wp[1]-pos[1])
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

            front_dist = min(ps_to_dist(prox_readings[0]), ps_to_dist(prox_readings[7]))

            THETA_HIGH = math.radians(30)
            THETA_LOW  = math.radians(8)

            if abs(heading_error) > THETA_HIGH:
                _is_pivoting = True
            elif _is_pivoting and abs(heading_error) < THETA_LOW:
                _is_pivoting = False

            if _is_pivoting:
                base_velocity = 0.0
            else:
                base_velocity = 3.0
                if dist_to_wp < 0.40:
                    base_velocity = max(0.5, base_velocity * (dist_to_wp / 0.40))
                if front_dist < 0.40:
                    base_velocity *= (front_dist / 0.40)

            angular_velocity = np.clip(heading_error * 7.5, -4.0, 4.0)

        # ── ALGORITHM: PPO (trained MultiInputPolicy) ─────────────────────────
        elif NAVIGATION_MODE == "PPO":
            obs_dict = _build_ppo_obs(pos, prox_readings, current_yaw,
                                      current_lane_index)
            action, _ = ppo_model.predict(obs_dict, deterministic=True)
            # Map MistNavEnv action convention → wheel velocities directly,
            # using the SAME scaling constants as training (imported above).
            avg  = float(action[0]) * FORWARD_SCALE
            diff = float(action[1]) * TURN_SCALE
            wheels[0].setVelocity(np.clip(avg + diff, -MAX_SPEED, MAX_SPEED))
            wheels[1].setVelocity(np.clip(avg - diff, -MAX_SPEED, MAX_SPEED))
            continue   # wheel mixing is handled above; skip the shared block below

        else:
            base_velocity    = 0.0
            angular_velocity = 0.0

        # ── Wheel mixing (APF + D*; PPO already set velocities via continue) ──
        # Standard differential drive: left faster → CCW (positive yaw)
        wheels[0].setVelocity(np.clip(base_velocity + angular_velocity, -MAX_SPEED, MAX_SPEED))
        wheels[1].setVelocity(np.clip(base_velocity - angular_velocity, -MAX_SPEED, MAX_SPEED))

except Exception as e:
    print(f"❌ CRITICAL RUNTIME EXCEPTION: {e}")
    raise e

finally:
    print("💾 Closing performance data streams.")
    csv_file.close()
    robot.simulationQuit(0)