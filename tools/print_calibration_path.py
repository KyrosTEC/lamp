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
        print("Calibration path:")
        print(robot.calibration_fpath)

        print("\nCalibration dir:")
        print(robot.calibration_dir)

        print("\nCurrent calibration:")
        print(robot.calibration)

    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()