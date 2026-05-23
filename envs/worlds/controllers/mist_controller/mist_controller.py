from controller import Supervisor
import numpy as np
import math
import sys

# ==========================================
# 1. CORE INITIALIZATION & PRIVILEGE CHECKS
# ==========================================
robot = Supervisor()
timestep = int(robot.getBasicTimeStep())

robot_node = robot.getSelf()
if robot_node is None:
    print("❌ SYSTEM CRITICAL: robot.getSelf() returned None! Grant supervisor privileges.")
    sys.exit(1)

# ==========================================
# 2. AUTOMATED DEVICE RECOVERY MATRIX
# ==========================================
lidar = None
camera = None
device_count = robot.getNumberOfDevices()

for i in range(device_count):
    device = robot.getDeviceByIndex(i)
    class_name = device.__class__.__name__
    if class_name == "Lidar" and lidar is None:
        lidar = device
    elif class_name == "Camera" and camera is None:
        camera = device

if lidar:
    lidar.enable(timestep)
else:
    print("❌ SYSTEM CRITICAL: Scanning LiDAR could not be found.")
    sys.exit(1)

if camera:
    camera.enable(timestep)

wheels = [robot.getDevice("left wheel motor"), robot.getDevice("right wheel motor")]
for w in wheels:
    w.setPosition(float('inf'))
    w.setVelocity(0.0)

# ==========================================
# 3. ENVIRONMENT & KINEMATIC MATRIX SETTINGS
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

# Physical structural constraints
MAX_SPEED = 6.28
ROBOT_RADIUS = 0.037


def reset_to_lane(lane_idx):
    """Teleports the e-puck to the beginning of the selected lane."""
    start_pos = START_ARRAY[lane_idx]
    print(f"🚀 TELEPORTING E-PUCK TO LANE {lane_idx} -> Coordinates: {start_pos}")
    robot_node.getField("translation").setSFVec3f(start_pos)
    robot_node.getField("rotation").setSFRotation(START_ROT)
    robot_node.resetPhysics()
    for w in wheels:
        w.setVelocity(0.0)


reset_to_lane(current_lane_index)

# Calibration Delay Warm-up Phase
start_time = robot.getTime()
while robot.step(timestep) != -1:
    if robot.getTime() - start_time >= 1.0:
        break

H_RES = lidar.getHorizontalResolution()  # Catches 111 point array
half_res = H_RES // 2
DEG_PER_INDEX = 360.0 / H_RES

print(f"\n🤖 >>> APF CONTROLLER ONLINE <<<")
print(f"📌 Active Resolution Profile: {H_RES} points | Step Delta: {DEG_PER_INDEX:.2f}°\n")

# ==========================================
# 4. ARTIFICIAL POTENTIAL FIELD GAINS
# ==========================================
K_ATTRACTIVE = 2.5  # Pull scaling factor towards the goal
K_REPULSIVE = 0.08  # Push scaling factor away from walls
INFLUENCE_DIST = 0.45  # Distance threshold where walls begin repelling the robot (d_0)
WINDOW_SIZE = 2  # Sliding window radius to find local minima (TP5 Todo 2a)

# ==========================================
# 5. MAIN NAVIGATION CONTROLLER LOOP
# ==========================================
while robot.step(timestep) != -1:
    pos = robot_node.getPosition()
    curr_time = robot.getTime()

    if camera:
        camera.getImage()  # Keep camera view active

    # Track Sequence/Goal Line Checker
    active_goal_x = GOAL_ARRAY[current_lane_index][0]
    if pos[0] >= active_goal_x:
        print(f"🏁 GOAL MET: Lane {current_lane_index} cleared successfully!")
        current_lane_index = (current_lane_index + 1) % num_lanes
        reset_to_lane(current_lane_index)
        continue

    # Get accurate robot orientation (Yaw)
    rot_matrix = robot_node.getOrientation()
    current_yaw = math.atan2(rot_matrix[3], rot_matrix[0])

    # Fetch LiDAR profile
    lidar_raw = lidar.getRangeImage()
    if lidar_raw is None:
        continue
    horizon = np.array(lidar_raw)
    horizon[np.isinf(horizon)] = 3.0
    horizon[horizon <= 0.05] = 3.0

    # ----------------------------------------------------------------
    # APF MATHEMATICAL ENGINE
    # ----------------------------------------------------------------

    # A. COMPUTE ATTRACTIVE FORCE VECTOR (Lecture 5, Slide 29)
    # Target coordinate point in world space
    target_pos = np.array([GOAL_ARRAY[current_lane_index][0], GOAL_ARRAY[current_lane_index][1]])
    robot_pos = np.array([pos[0], pos[1]])

    # Direct distance vector to goal
    vec_to_goal = target_pos - robot_pos
    dist_to_goal = np.linalg.norm(vec_to_goal)

    if dist_to_goal > 0:
        # Linear parabolic model scaling uniform pull force
        f_att = K_ATTRACTIVE * (vec_to_goal / dist_to_goal)
    else:
        f_att = np.array([0.0, 0.0])

    # B. COMPUTE REPULSIVE FORCE VECTOR (Lecture 5, Slide 34)
    f_rep = np.array([0.0, 0.0])

    for idx in range(H_RES):
        dist_reading = horizon[idx]

        # 1. Sliding Window Local Minima Filter (TP5 Todo 2a)
        is_local_minimum = True
        for offset in range(-WINDOW_SIZE, WINDOW_SIZE + 1):
            neighbor_idx = (idx + offset) % H_RES
            if horizon[neighbor_idx] < dist_reading:
                is_local_minimum = False
                break

        if not is_local_minimum:
            continue  # Skip calculations if this point isn't the apex of a wall cluster

        # 2. Distance Obstacle Field Check (Only compute within d_0 boundary)
        if dist_reading < INFLUENCE_DIST and dist_reading > 0.01:
            # Map index back to true 2D local polar coordinate angle
            if idx <= half_res:
                ray_angle_rad = math.radians(idx * DEG_PER_INDEX)
            else:
                ray_angle_rad = math.radians((idx - H_RES) * DEG_PER_INDEX)

            # Convert local laser coordinate space to global world coordinate frame vectors
            global_ray_angle = current_yaw + ray_angle_rad

            # Unit direction vector pointing away from the detected wall point
            obstacle_vector_x = -math.cos(global_ray_angle)
            obstacle_vector_y = -math.sin(global_ray_angle)
            vec_away_from_wall = np.array([obstacle_vector_x, obstacle_vector_y])

            # Standard Repulsive Force Function (Slide 34 Formulations):
            # F_rep = K * (1/d - 1/d_0) * (1/d^2) * DirectionVector
            factor = (1.0 / dist_reading) - (1.0 / INFLUENCE_DIST)
            magnitude = K_REPULSIVE * factor * (1.0 / (dist_reading ** 2))

            f_rep += magnitude * vec_away_from_wall

    # C. COMBINE FORCES TO COMPUTE RESULTING FORCE VECTOR
    f_total = f_att + f_rep

    # ----------------------------------------------------------------
    # KINEMATIC MOTION PLANNER (Lecture 5, Slide 38)
    # ----------------------------------------------------------------
    # Convert global force vector back to target tracking coordinates
    desired_heading_global = math.atan2(f_total[1], f_total[0])

    # Relative target tracking heading error calculation
    heading_error_rad = desired_heading_global - current_yaw
    heading_error_rad = (heading_error_rad + math.pi) % (2 * math.pi) - math.pi

    # Dynamic scaling mapping forces to differential motor speed parameters
    force_magnitude = np.linalg.norm(f_total)

    # Scale forward speed down proportional to severe angular corrections or strong wall pushback
    base_velocity = np.clip(force_magnitude * 1.2, 1.0, 4.0)
    if abs(heading_error_rad) > 0.6:
        base_velocity *= 0.3  # Slow down to spin smoothly if facing a harsh angle offset

    # Proportional angular correction
    angular_velocity = np.clip(heading_error_rad * 3.5, -2.5, 2.5)

    # Differential drive mixer transforms
    v_l = base_velocity - angular_velocity
    v_r = base_velocity + angular_velocity

    wheels[0].setVelocity(np.clip(v_l, -MAX_SPEED, MAX_SPEED))
    wheels[1].setVelocity(np.clip(v_r, -MAX_SPEED, MAX_SPEED))