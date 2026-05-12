from lerobot.robots.utils import make_robot_from_config
from lerobot.robots.so_follower import SO101FollowerConfig
import time

cfg = SO101FollowerConfig(
    port="/dev/ttyACM0",
    id="follower_kyros",
    max_relative_target=5.0,
)

robot = make_robot_from_config(cfg)

try:
    robot.connect()
    print("Conectado")

    for i in range(20):
        obs = robot.get_observation()
        print(i, obs)
        time.sleep(0.5)

finally:
    try:
        robot.disconnect()
    except Exception:
        pass