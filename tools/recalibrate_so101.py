from lerobot.robots.utils import make_robot_from_config
from lerobot.robots.so_follower import SO101FollowerConfig


PORT = "/dev/ttyACM0"
ROBOT_ID = "follower_kyros"


def main():
    cfg = SO101FollowerConfig(
        port=PORT,
        id=ROBOT_ID,
        max_relative_target=90.0,
    )

    robot = make_robot_from_config(cfg)
    robot.connect()

    try:
        print("Iniciando calibración del SO101...")
        print("Sigue las instrucciones que aparezcan en terminal.")
        print("Mueve cada articulación por TODO su rango físico seguro.")
        print()
        print("IMPORTANTE:")
        print("- Mueve bien shoulder_pan, o sea la base.")
        print("- Mueve bien shoulder_lift.")
        print("- Mueve wrist_flex.")
        print("- Abre/cierra gripper si aplica.")
        print("- No fuerces el brazo contra topes mecánicos.")

        robot.calibrate()

        print("\nCalibración terminada.")
        print("Nueva calibración:")
        print(robot.calibration)

    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()