import time
from pprint import pprint

from lerobot.robots.utils import make_robot_from_config
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


JOINT_KEYS = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]


ZONE_NAMES = {
    1: "ARRIBA IZQUIERDA",
    2: "ARRIBA CENTRO",
    3: "ARRIBA DERECHA",
    4: "CENTRO IZQUIERDA",
    5: "CENTRO",
    6: "CENTRO DERECHA",
    7: "ABAJO IZQUIERDA",
    8: "ABAJO CENTRO",
    9: "ABAJO DERECHA",
}


class SO101CalibrationController:
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
        observation = self.robot.get_observation()

        pose = {}

        for key in JOINT_KEYS:
            if key in observation:
                pose[key] = float(observation[key])
            else:
                print(f"Advertencia: no se encontró la llave {key}")

        return pose

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

    def disable_torque(self):
        """
        Intenta desactivar torque para poder mover el brazo manualmente.

        Dependiendo de la versión de LeRobot/Feetech, el método puede cambiar.
        Por eso se prueban varias opciones comunes.
        """

        print("Intentando desactivar torque...")

        possible_targets = [
            self.robot,
            getattr(self.robot, "bus", None),
            getattr(self.robot, "motors_bus", None),
            getattr(self.robot, "follower_arm", None),
        ]

        for target in possible_targets:
            if target is None:
                continue

            for method_name in [
                "disable_torque",
                "disable_motors",
                "torque_disable",
                "set_torque_off",
            ]:
                if hasattr(target, method_name):
                    method = getattr(target, method_name)
                    try:
                        method()
                        print(f"Torque desactivado usando: {target}.{method_name}()")
                        return True
                    except TypeError:
                        pass
                    except Exception as e:
                        print(f"No funcionó {method_name}: {e}")

        print("No encontré un método directo para desactivar torque.")
        return False

    def enable_torque(self):
        """
        Intenta activar torque nuevamente.
        """

        print("Intentando activar torque...")

        possible_targets = [
            self.robot,
            getattr(self.robot, "bus", None),
            getattr(self.robot, "motors_bus", None),
            getattr(self.robot, "follower_arm", None),
        ]

        for target in possible_targets:
            if target is None:
                continue

            for method_name in [
                "enable_torque",
                "enable_motors",
                "torque_enable",
                "set_torque_on",
            ]:
                if hasattr(target, method_name):
                    method = getattr(target, method_name)
                    try:
                        method()
                        print(f"Torque activado usando: {target}.{method_name}()")
                        return True
                    except TypeError:
                        pass
                    except Exception as e:
                        print(f"No funcionó {method_name}: {e}")

        print("No encontré un método directo para activar torque.")
        return False

    def print_available_methods(self):
        """
        Ayuda para encontrar cómo se llama el método correcto
        en tu versión de LeRobot.
        """

        print("\nMétodos disponibles relacionados con torque/motor/bus:")
        targets = {
            "robot": self.robot,
            "robot.bus": getattr(self.robot, "bus", None),
            "robot.motors_bus": getattr(self.robot, "motors_bus", None),
            "robot.follower_arm": getattr(self.robot, "follower_arm", None),
        }

        for name, target in targets.items():
            if target is None:
                continue

            print(f"\n{name}:")
            methods = dir(target)

            for m in methods:
                lower = m.lower()
                if (
                    "torque" in lower
                    or "motor" in lower
                    or "bus" in lower
                    or "connect" in lower
                    or "write" in lower
                ):
                    print(f"  - {m}")


def write_zone_poses_file(zone_poses, filename="zone_poses.py"):
    with open(filename, "w") as file:
        file.write("# Archivo generado automáticamente por calibrate_zone_poses.py\n\n")

        file.write("ZONE_POSES = {\n")

        for zone, pose in zone_poses.items():
            file.write(f"    {zone}: {{\n")

            for joint, value in pose.items():
                file.write(f'        "{joint}": {value},\n')

            file.write("    },\n")

        file.write("}\n\n")

        file.write("ZONE_NAMES = {\n")
        for zone, name in ZONE_NAMES.items():
            file.write(f'    {zone}: "{name}",\n')
        file.write("}\n")

    print(f"\nArchivo generado: {filename}")


def main():
    controller = SO101CalibrationController()
    zone_poses = {}

    print("==========================================")
    print(" Calibrador de posiciones por zona SO-101")
    print("==========================================")
    print()
    print("Zonas:")
    print("  1 | 2 | 3")
    print("  4 | 5 | 6")
    print("  7 | 8 | 9")
    print()
    print("IMPORTANTE:")
    print("Cuando el torque se desactive, el brazo puede caer.")
    print("Sujétalo con la mano antes de confirmar.")
    print()

    try:
        controller.connect()

        print("\nPrimero revisaremos si podemos desactivar torque.")
        torque_disabled = controller.disable_torque()

        if not torque_disabled:
            controller.print_available_methods()
            print()
            print("No se pudo desactivar el torque automáticamente.")
            print("Alternativa rápida:")
            print("1. Cierra este script.")
            print("2. Ejecuta: python -m src.tools.servo_disable")
            print("3. Vuelve a correr este calibrador.")
            return

        print()
        print("Torque desactivado.")
        print("Ahora deberías poder mover el brazo manualmente.")
        print()

        for zone in [5, 2, 8, 4, 6, 1, 3, 7, 9]:
            print()
            print("------------------------------------------")
            print(f"Calibrando zona {zone}: {ZONE_NAMES[zone]}")
            print("------------------------------------------")
            print("Mueve manualmente el brazo hasta que apunte a esta zona.")
            print("Cuando esté listo, presiona ENTER.")
            print("También puedes escribir:")
            print("  skip = saltar zona")
            print("  q    = salir")
            print()

            user_input = input(f"Guardar pose para zona {zone}? ").strip().lower()

            if user_input == "q":
                print("Saliendo de calibración.")
                break

            if user_input == "skip":
                print(f"Zona {zone} saltada.")
                continue

            pose = controller.get_pose()
            zone_poses[zone] = pose

            print(f"Pose guardada para zona {zone}: {ZONE_NAMES[zone]}")
            pprint(pose)

        if zone_poses:
            print()
            print("==========================================")
            print("Resumen de poses guardadas")
            print("==========================================")
            pprint(zone_poses)

            write_zone_poses_file(zone_poses)
        else:
            print("No se guardó ninguna pose.")

    finally:
        print("\nFinalizando calibración.")
        print("Intentando activar torque antes de desconectar...")
        try:
            controller.enable_torque()
        except Exception as e:
            print(f"No se pudo activar torque al final: {e}")

        controller.disconnect()


if __name__ == "__main__":
    main()