import time
from lerobot.robots import make_robot_from_config
from lerobot.robots.so_follower import SO101FollowerConfig


PORT = "/dev/ttyACM0"
ROBOT_ID = "follower_kyros"


HOME_POSE = {
    "shoulder_pan.pos": -0.6593406593406593,
    "shoulder_lift.pos": -104.17582417582418,
    "elbow_flex.pos": 96.48351648351648,
    "wrist_flex.pos": 71.91208791208791,
    "wrist_roll.pos": -89.53846153846153,
    "gripper.pos": 8.83785664578984,
}

READY_POSE = {
    "shoulder_pan.pos": -15.868131868131869,
    "shoulder_lift.pos": -65.58241758241758,
    "elbow_flex.pos": 5.318681318681318,
    "wrist_flex.pos": 80.26373626373626,
    "wrist_roll.pos": -82.24175824175825,
    "gripper.pos": 4.453723034098817,
}


class SO101Controller:
    def __init__(self):
        cfg = SO101FollowerConfig(
            port=PORT,
            id=ROBOT_ID,
            max_relative_target=10.0,
        )

        self.robot = make_robot_from_config(cfg)
        self.connected = False

    def connect(self):
        self.robot.connect()
        self.connected = True
        print("SO-101 conectado.")

    def disconnect(self):
        if self.connected:
            self.robot.disconnect()
            self.connected = False
            print("SO-101 desconectado.")

    def get_pose(self):
        return self.robot.get_observation()

    def smooth_move_to_pose(self, target_pose, duration=2.0, steps=80):
        start = self.robot.get_observation()

        for i in range(steps + 1):
            alpha = i / steps
            action = start.copy()

            for joint, target_value in target_pose.items():
                if joint in start:
                    action[joint] = start[joint] + alpha * (target_value - start[joint])

            self.robot.send_action(action)
            time.sleep(duration / steps)

    def go_home(self):
        print("Moviendo a HOME...")
        self.smooth_move_to_pose(HOME_POSE, duration=2.5, steps=100)

    def go_ready(self):
        print("Moviendo a READY...")
        self.smooth_move_to_pose(READY_POSE, duration=2.0, steps=80)