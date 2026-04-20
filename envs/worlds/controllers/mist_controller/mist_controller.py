from controller import Robot, Camera, Lidar
import random

def run_robot():
    # 1. Initialize the Robot
    robot = Robot()
    timestep = int(robot.getBasicTimeStep())

    # 2. Initialize and Enable Camera
    # The name "camera" must match the name in your Webots Scene Tree
    cam = robot.getDevice("camera")
    cam.enable(timestep)

    # 3. Initialize and Enable LiDAR
    # Ensure you added a Lidar node to the extensionSlot named "lidar"
    lidar = robot.getDevice("lidar")
    if lidar:
        lidar.enable(timestep)
    else:
        print("Lidar not found! Check the name in the Scene Tree.")

    # 4. Get Motor Devices (Pioneer 3-AT uses 4 motors)
    left_front = robot.getDevice("left front wheel")
    left_rear = robot.getDevice("left back wheel")
    right_front = robot.getDevice("right front wheel")
    right_rear = robot.getDevice("right back wheel")

    # Set them to velocity mode
    for motor in [left_front, left_rear, right_front, right_rear]:
        motor.setPosition(float('inf'))
        motor.setVelocity(0.0)

    print("--- MIST Controller Started ---")

    # Main Control Loop
    while robot.step(timestep) != -1:
        # --- SENSOR PROCESSING ---
        
        # Get LiDAR data
        if lidar:
            raw_lidar_values = lidar.getRangeImage()
            
            # SIMULATING MIST: Adding Gaussian Noise
            # Increase 'sigma' to make the robot more "blind"
            sigma = 0.05 
            noisy_lidar = [val + random.gauss(0, sigma) for val in raw_lidar_values]
            
            # For debugging: print the distance to the object directly in front
            # (Center of the array)
            mid = len(noisy_lidar) // 2
            # print(f"Noisy Distance Ahead: {noisy_lidar[mid]:.2f}m")

        # --- BASIC MOVEMENT (Just for testing) ---
        # Let's make it move forward at a slow speed
        speed = 2.0
        left_front.setVelocity(speed)
        left_rear.setVelocity(speed)
        right_front.setVelocity(speed)
        right_rear.setVelocity(speed)

if __name__ == "__main__":
    run_robot()