import csv
import os
from datetime import datetime


JOINT_KEYS = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]


CSV_FIELDNAMES = [
    "timestamp",
    "iteration",
    "zone",
    "shoulder_pan_real",
    "shoulder_lift_real",
    "elbow_flex_real",
    "wrist_flex_real",
    "wrist_roll_real",
    "gripper_real",
    "shoulder_pan_cmd",
    "shoulder_lift_cmd",
    "elbow_flex_cmd",
    "wrist_flex_cmd",
    "wrist_roll_cmd",
    "gripper_cmd",
    "shoulder_pan_error",
    "shoulder_lift_error",
    "elbow_flex_error",
    "wrist_flex_error",
    "wrist_roll_error",
    "gripper_error",
    "max_error",
]


CSV_KEY_MAP = {
    "shoulder_pan.pos": "shoulder_pan",
    "shoulder_lift.pos": "shoulder_lift",
    "elbow_flex.pos": "elbow_flex",
    "wrist_flex.pos": "wrist_flex",
    "wrist_roll.pos": "wrist_roll",
    "gripper.pos": "gripper",
}


class MPCLogger:
    def __init__(self, logs_dir="logs"):
        self.logs_dir = logs_dir
        self.filepath = None
        self.file = None
        self.writer = None

        os.makedirs(self.logs_dir, exist_ok=True)

    def start(self, label="mpc_run"):
        safe_label = str(label).replace(" ", "_").replace("/", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        filename = f"{timestamp}_{safe_label}.csv"
        self.filepath = os.path.join(self.logs_dir, filename)

        self.file = open(self.filepath, mode="w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=CSV_FIELDNAMES)
        self.writer.writeheader()

        print(f"MPC logger iniciado: {self.filepath}")

        return self.filepath

    def log_iteration(
        self,
        iteration,
        zone,
        real_pose,
        command_pose,
        errors,
        max_error,
    ):
        if self.writer is None:
            return

        row = {
            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            "iteration": iteration,
            "zone": zone,
            "max_error": max_error,
        }

        for joint in JOINT_KEYS:
            base_key = CSV_KEY_MAP[joint]

            row[f"{base_key}_real"] = real_pose.get(joint, "")
            row[f"{base_key}_cmd"] = command_pose.get(joint, "")
            row[f"{base_key}_error"] = errors.get(joint, "")

        self.writer.writerow(row)

    def close(self):
        if self.file is not None:
            self.file.flush()
            self.file.close()

            print(f"MPC logger cerrado: {self.filepath}")

        self.file = None
        self.writer = None