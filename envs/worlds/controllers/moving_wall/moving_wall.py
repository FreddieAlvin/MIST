import random
from controller import Robot, Node  # Imported Node here


def main():
    # 1. Initialize the Robot instance
    robot = Robot()
    timestep = int(robot.getBasicTimeStep())

    # 2. Pioneer 3-AT 4-Wheel Motor Map
    left_motors = []
    right_motors = []

    device_count = robot.getNumberOfDevices()
    for i in range(device_count):
        device = robot.getDeviceByIndex(i)

        # FIXED: Webots uses Node.ROTATIONAL_MOTOR and Node.LINEAR_MOTOR
        node_type = device.getNodeType()
        if node_type in [Node.ROTATIONAL_MOTOR, Node.LINEAR_MOTOR]:
            motor_name = device.getName().lower()

            # Map motors to their respective sides
            if "left" in motor_name or motor_name.startswith("l"):
                left_motors.append(device)
            elif "right" in motor_name or motor_name.startswith("r"):
                right_motors.append(device)

    # Verify we found the P3-AT motors
    all_motors = left_motors + right_motors
    if not all_motors:
        print("❌ Error: Could not find any motor devices on this Pioneer 3-AT.")
        return

    print(f"✅ Hardware Found: {len(left_motors)} Left Motors, {len(right_motors)} Right Motors.")

    # Configure all found motors for infinite velocity mode
    for motor in all_motors:
        motor.setPosition(float('inf'))
        motor.setVelocity(0.0)

    # 3. Define movement configurations
    CRUISE_SPEED = 2.0
    LEG_DURATION = 4.0  # Seconds spent driving forward/backward before swapping

    # 4. RANDOM INITIALIZATION DELAY (Desynchronization Step)
    # Usamos o nome do robô para garantir que a semente é verdadeiramente única!
    random.seed(hash(robot.getName()))
    start_delay = random.uniform(0.1, 5.0)
    print(f"🤖 [{robot.getName()}] Initialized. Waiting {start_delay:.2f} seconds.")

    # Loop de espera seguro
    delay_start_time = robot.getTime()
    while robot.step(timestep) != -1:
        # Se o tempo do Webots avançar além do delay, avançamos
        if robot.getTime() - delay_start_time >= start_delay:
            break
    # Wait out the random initialization delay while keeping motors dead still
    delay_start_time = robot.getTime()
    while robot.step(timestep) != -1:
        if robot.getTime() - delay_start_time >= start_delay:
            break

    print(f"🚀 [{robot.getName()}] Delay concluded. Starting P3-AT back-and-forth loop.")

    # 5. Main Movement Cycle Loop
    direction_multiplier = 1
    leg_start_time = robot.getTime()

    while robot.step(timestep) != -1:
        current_time = robot.getTime()

        # Check if the robot has finished its current straight leg duration
        if current_time - leg_start_time >= LEG_DURATION:
            direction_multiplier *= -1
            leg_start_time = current_time

        # Compute target velocity
        target_velocity = CRUISE_SPEED * direction_multiplier

        # Drive all left motors
        for motor in left_motors:
            motor.setVelocity(target_velocity)

        # Drive all right motors
        for motor in right_motors:
            motor.setVelocity(target_velocity)


if __name__ == "__main__":
    main()