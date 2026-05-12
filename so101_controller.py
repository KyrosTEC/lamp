import time
from typing import Dict, Optional

from lerobot.robots.utils import make_robot_from_config
from lerobot.robots.so_follower import SO101FollowerConfig

from zone_poses import ZONE_POSES, ZONE_NAMES


PORT = "/dev/ttyACM0"
ROBOT_ID = "follower_kyros"

Pose = Dict[str, float]


HOME_POSE: Pose = {
    "shoulder_pan.pos": -0.8351648351648352,
    "shoulder_lift.pos": -89.0989010989011,
    "elbow_flex.pos": -0.04395604395604396,
    "wrist_flex.pos": 65.67032967032966,
    "wrist_roll.pos": -10.68131868131868,
    "gripper.pos": 85.71428571428571,
}

READY_POSE: Pose = {
    "shoulder_pan.pos": -15.868131868131869,
    "shoulder_lift.pos": -65.58241758241758,
    "elbow_flex.pos": 5.318681318681318,
    "wrist_flex.pos": 80.26373626373626,
    "wrist_roll.pos": -82.24175824175825,
    "gripper.pos": 4.453723034098817,
}

# Por ahora tu HOME ya es segura, entonces SAFE_HOME usa la misma pose.
SAFE_HOME_POSE: Pose = HOME_POSE.copy()

# Si más adelante quieres regresar con una trayectoria más alta/intermedia,
# puedes cambiar esta pose sin tocar SAFE_HOME_POSE.
SAFE_TRANSITION_POSE: Optional[Pose] = None


JOINT_KEYS = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]


class SO101Controller:
    def __init__(self):
        cfg = SO101FollowerConfig(
            port=PORT,
            id=ROBOT_ID,
            max_relative_target=90.0,
        )

        self.robot = make_robot_from_config(cfg)
        self.connected = False

    def connect(self):
        if self.connected:
            print("SO-101 ya estaba conectado.")
            return

        self.robot.connect()
        self.connected = True

        print("SO-101 conectado.")
        self.print_current_pose("Pose actual al conectar")

    def disconnect(self):
        if not self.connected:
            return

        self.robot.disconnect()
        self.connected = False
        print("SO-101 desconectado.")

    def get_pose(self):
        return self.robot.get_observation()

    def print_current_pose(self, title="Pose actual"):
        print(f"\n{title}:")
        obs = self.get_pose()

        for joint in JOINT_KEYS:
            if joint in obs:
                print(f"  {joint}: {obs[joint]}")
            else:
                print(f"  WARNING: no existe {joint} en observation")

    def validate_pose(self, target_pose: Pose) -> bool:
        """
        Valida que la pose tenga al menos un joint válido.
        No bloquea si falta alguno, solo avisa.
        """

        if not target_pose:
            print("ERROR: La pose está vacía.")
            return False

        current_pose = self.get_pose()
        valid_joints = 0

        for joint in target_pose:
            if joint not in JOINT_KEYS:
                print(f"WARNING: {joint} no está en JOINT_KEYS.")

            if joint not in current_pose:
                print(f"WARNING: {joint} no existe en la observación actual.")
            else:
                valid_joints += 1

        if valid_joints == 0:
            print("ERROR: Ningún joint de la pose existe en la observación actual.")
            return False

        return True

    def smooth_move_to_pose(self, target_pose: Pose, duration=2.0, steps=100):
        """
        Mueve el robot suavemente desde la pose actual hasta target_pose.

        duration: duración total del movimiento en segundos.
        steps: cantidad de interpolaciones.
        """

        if not self.connected:
            print("ERROR: El robot no está conectado.")
            return

        if not self.validate_pose(target_pose):
            return

        start_pose = self.get_pose()

        print("\n==========================================")
        print("MOVIMIENTO A POSE")
        print("==========================================")

        print("Diferencias detectadas:")
        for joint in JOINT_KEYS:
            if joint in target_pose and joint in start_pose:
                diff = target_pose[joint] - start_pose[joint]
                print(
                    f"  {joint}: "
                    f"{start_pose[joint]:.3f} -> {target_pose[joint]:.3f} "
                    f"diff={diff:.3f}"
                )

        if steps <= 0:
            steps = 1

        delay = duration / steps

        for i in range(steps + 1):
            alpha = i / steps
            action = start_pose.copy()

            for joint in JOINT_KEYS:
                if joint in start_pose and joint in target_pose:
                    start_value = start_pose[joint]
                    target_value = target_pose[joint]
                    action[joint] = start_value + alpha * (target_value - start_value)

            self.robot.send_action(action)
            time.sleep(delay)

        self.print_current_pose("Pose final leída")

    def go_home(self):
        print("\nMoviendo a HOME...")
        self.smooth_move_to_pose(
            HOME_POSE,
            duration=2.5,
            steps=120,
        )

    def go_ready(self):
        print("\nMoviendo a READY...")
        self.smooth_move_to_pose(
            READY_POSE,
            duration=2.0,
            steps=100,
        )

    def go_safe_home(self):
        """
        Regresa a una posición segura.

        Si SAFE_TRANSITION_POSE existe, primero pasa por ella.
        Si no existe, va directo a SAFE_HOME_POSE.
        """

        print("\n==========================================")
        print("MOVIENDO A SAFE HOME")
        print("==========================================")

        if SAFE_TRANSITION_POSE is not None:
            print("Paso 1/2: moviendo a pose de transición segura...")
            self.smooth_move_to_pose(
                SAFE_TRANSITION_POSE,
                duration=2.5,
                steps=120,
            )

            print("Paso 2/2: moviendo a HOME segura...")
            self.smooth_move_to_pose(
                SAFE_HOME_POSE,
                duration=2.5,
                steps=120,
            )
        else:
            print("Moviendo directo a HOME segura...")
            self.smooth_move_to_pose(
                SAFE_HOME_POSE,
                duration=2.5,
                steps=120,
            )

        print("Robot en SAFE HOME.")

    def go_to_zone(self, zone: int):
        if zone not in ZONE_POSES:
            print(f"No existe pose para zona {zone}.")
            return

        zone_name = ZONE_NAMES.get(zone, "ZONA DESCONOCIDA")
        target_pose = ZONE_POSES[zone]

        print("\n==========================================")
        print(f"MOVIENDO A ZONA {zone}: {zone_name}")
        print("==========================================")

        print("Valores objetivo importantes:")
        print(f"  shoulder_pan.pos: {target_pose.get('shoulder_pan.pos')}")
        print(f"  shoulder_lift.pos: {target_pose.get('shoulder_lift.pos')}")

        self.smooth_move_to_pose(
            target_pose,
            duration=2.0,
            steps=120,
        )