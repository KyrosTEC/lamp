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
        print("\n===================================")
        print("OBSERVATION")
        print("===================================")
        obs = robot.get_observation()

        for key, value in obs.items():
            print(f"{key}: {value}")

        print("\n===================================")
        print("ROBOT ATTRIBUTES")
        print("===================================")

        for attr in dir(robot):
            lower = attr.lower()
            if (
                "calib" in lower
                or "motor" in lower
                or "bus" in lower
                or "config" in lower
                or "joint" in lower
            ):
                print(attr)

        print("\n===================================")
        print("CALIBRATION / CONFIG POSSIBLE DATA")
        print("===================================")

        possible_attrs = [
            "calibration",
            "calibration_data",
            "calib",
            "config",
            "robot_type",
            "motors",
            "bus",
            "motors_bus",
        ]

        for attr in possible_attrs:
            if hasattr(robot, attr):
                print(f"\n--- robot.{attr} ---")
                try:
                    print(getattr(robot, attr))
                except Exception as e:
                    print(f"No se pudo imprimir {attr}: {e}")

    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()