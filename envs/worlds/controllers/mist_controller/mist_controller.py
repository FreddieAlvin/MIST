from controller import Supervisor
import numpy as np

# --- INITIALIZATION ---
robot = Supervisor()
timestep = int(robot.getBasicTimeStep())
robot_node = robot.getSelf()

# Sensors
camera = robot.getDevice("camera")
if camera: camera.enable(timestep)
lidar = robot.getDevice("lidar")
if lidar: lidar.enable(timestep)

# Motors
wheels = [robot.getDevice(n) for n in ['front left wheel', 'front right wheel', 'back left wheel', 'back right wheel']]
for w in wheels:
    w.setPosition(float('inf'))
    w.setVelocity(0.0)

# --- CONFIG ---
START_POS = [-10.8, 3.13, 0.15]
START_ROT = [0, 0, 1, 0]  # Face East
GOAL_X = 10.75

# Thresholds
MAX_SPEED = 6.4
SLOW_DIST = 4.0  # Start slowing down 4 meters away
STOP_DIST = 0.7  # STOP/Reverse at 0.7 meters

# --- FORCE START POSITION ---
robot_node.getField("translation").setSFVec3f(START_POS)
robot_node.getField("rotation").setSFRotation(START_ROT)
robot_node.resetPhysics()

# 1-sec wait
start_time = robot.getTime()
while robot.step(timestep) != -1:
    if robot.getTime() - start_time >= 1.0: break

# --- MAIN LOOP ---
while robot.step(timestep) != -1:
    pos = robot_node.getPosition()

    if pos[0] >= GOAL_X:
        for w in wheels: w.setVelocity(0.0)
        print(">>> GOAL REACHED")
        break

    # 1. PERCEPTION
    lidar_raw = lidar.getRangeImage()
    if not lidar_raw: continue

    scan = np.array(lidar_raw)
    scan[np.isinf(scan)] = 10.0
    scan[scan <= 0.05] = 10.0  # Ignore internal noise

    mid = len(scan) // 2
    # Check 180-degree sweep to make sure we don't miss the yellow wall
    f_dist = np.min(scan[mid - 100:mid + 100])

    # 2. SPEED SCALING (The "Slow Down" Logic)
    if f_dist > SLOW_DIST:
        v = 2.0  # Cruise
    elif f_dist < STOP_DIST:
        v = -0.3  # Emergency Reverse if we hit the wall
    else:
        # Linear Speed scaling: Closer = Slower
        ratio = (f_dist - STOP_DIST) / (SLOW_DIST - STOP_DIST)
        v = ratio * 2.0

    # 3. STEERING (Desviar)
    # Check left vs right side gaps
    l_gap = np.mean(scan[mid:mid + 120])
    r_gap = np.mean(scan[mid - 120:mid])

    if f_dist < SLOW_DIST:
        # Hard turn away from the closest obstruction
        omega = 5.0 if l_gap > r_gap else -5.0
    else:
        # Nav: Try to stay on the center line (Y=3.13)
        omega = (3.13 - pos[1]) * 5.0

    # 4. FINAL DRIVE
    v_l = v - omega
    v_r = v + omega

    for i, w in enumerate(wheels):
        w.setVelocity(np.clip(v_l if i % 2 == 0 else v_r, -MAX_SPEED, MAX_SPEED))

    if int(robot.getTime() * 5) % 5 == 0:
        print(f"X: {pos[0]:.2f} | LiDAR_MIN: {f_dist:.2f} | Mode: {'AVOIDING' if f_dist < SLOW_DIST else 'CRUISING'}")