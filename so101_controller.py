import time
import math
from typing import Dict, Optional

# pyrefly: ignore [missing-import]
from lerobot.robots.utils import make_robot_from_config
# pyrefly: ignore [missing-import]
from lerobot.robots.so_follower import SO101FollowerConfig

from zone_poses import ZONE_POSES, ZONE_NAMES
from mpc_controller import MPCJointController
from mpc_logger import MPCLogger


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

SAFE_HOME_POSE: Pose = HOME_POSE.copy()

# Si quieres una trayectoria intermedia antes de HOME,
# cambia None por una pose real.
SAFE_TRANSITION_POSE: Optional[Pose] = None


JOINT_KEYS = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]


# ---------------------------------------------------------------------------
# Mapa de coordenadas normalizadas (x_norm, y_norm) para cada zona.
# Corresponde a la grilla 3x3: zona 1=arriba-izq ... zona 9=abajo-der.
# Se usa como centro de referencia para la interpolación IDW.
# ---------------------------------------------------------------------------
ZONE_GRID_COORDS = {
    1: (1/6, 1/6),   # arriba izquierda
    2: (3/6, 1/6),   # arriba centro
    3: (5/6, 1/6),   # arriba derecha
    4: (1/6, 3/6),   # centro izquierda
    5: (3/6, 3/6),   # centro
    6: (5/6, 3/6),   # centro derecha
    7: (1/6, 5/6),   # abajo izquierda
    8: (3/6, 5/6),   # abajo centro
    9: (5/6, 5/6),   # abajo derecha
}


class EMAFilter:
    """
    Filtro Exponential Moving Average para suavizar coordenadas ruidosas.

    y[n] = alpha * x[n] + (1 - alpha) * y[n-1]

    Alpha bajo (e.g. 0.2) = muy suave, más lag.
    Alpha alto (e.g. 0.5) = más reactivo, menos suave.
    """

    def __init__(self, alpha: float = 0.3):
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"EMAFilter alpha debe estar en (0, 1]. Valor recibido: {alpha}")
        self.alpha = alpha
        self._value: Optional[float] = None

    def update(self, x: float) -> float:
        if self._value is None:
            self._value = x
        else:
            self._value = self.alpha * x + (1.0 - self.alpha) * self._value
        return self._value

    def reset(self):
        self._value = None

    @property
    def value(self) -> Optional[float]:
        return self._value


class SO101Controller:
    def __init__(self):
        cfg = SO101FollowerConfig(
            port=PORT,
            id=ROBOT_ID,
            max_relative_target=90.0,
        )

        self.robot = make_robot_from_config(cfg)
        self.connected = False

        self.mpc = MPCJointController(
            horizon=12,
            max_step_deg=3.0,
            position_tolerance=2.0,
            control_gain=0.65,
        )

        self.logger = MPCLogger(logs_dir="logs")

        # Filtros EMA para suavizar las coordenadas normalizadas del objeto.
        self._ema_x = EMAFilter(alpha=0.3)
        self._ema_y = EMAFilter(alpha=0.3)

        # Pose interpolada más reciente (para tracking continuo).
        self._tracking_command_pose: Optional[Pose] = None

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

    # -----------------------------------------------------------------------
    # Interpolación IDW: convierte posición normalizada en pose articular
    # -----------------------------------------------------------------------

    def interpolate_pose_from_position(
        self,
        x_norm: float,
        y_norm: float,
        power: float = 2.0,
        smooth: bool = True,
    ) -> Optional[Pose]:
        """
        Interpola una pose articular usando Inverse Distance Weighting (IDW)
        entre las poses de zona calibradas.

        Args:
            x_norm: Coordenada X normalizada del objeto [0, 1].
            y_norm: Coordenada Y normalizada del objeto [0, 1].
            power:  Exponente IDW. Más alto = más peso a la zona más cercana.
            smooth: Si True, aplica filtro EMA antes de interpolar.

        Returns:
            Pose interpolada como dict, o None si no hay zonas calibradas.
        """
        if not ZONE_POSES:
            return None

        # Aplicar filtro EMA a las coordenadas del objeto
        if smooth:
            x_s = self._ema_x.update(x_norm)
            y_s = self._ema_y.update(y_norm)
        else:
            x_s = x_norm
            y_s = y_norm

        # Calcular distancias a cada zona en el espacio normalizado
        distances = {}
        for zone, (zx, zy) in ZONE_GRID_COORDS.items():
            if zone not in ZONE_POSES:
                continue
            dist = math.sqrt((x_s - zx) ** 2 + (y_s - zy) ** 2)
            distances[zone] = dist

        # Si el objeto cae exactamente sobre una zona, retornar esa pose
        for zone, dist in distances.items():
            if dist < 1e-9:
                return ZONE_POSES[zone].copy()

        # Calcular pesos IDW
        weights = {zone: 1.0 / (dist ** power) for zone, dist in distances.items()}
        total_weight = sum(weights.values())

        # Interpolar joint por joint
        interpolated: Pose = {}
        for joint in JOINT_KEYS:
            value = 0.0
            for zone, w in weights.items():
                if joint in ZONE_POSES[zone]:
                    value += w * ZONE_POSES[zone][joint]
            interpolated[joint] = value / total_weight

        return interpolated

    def reset_tracking_filters(self):
        """Reinicia los filtros EMA. Llamar al perder el objeto."""
        self._ema_x.reset()
        self._ema_y.reset()
        self._tracking_command_pose = None

    def print_current_pose(self, title="Pose actual"):
        print(f"\n{title}:")
        obs = self.get_pose()

        for joint in JOINT_KEYS:
            if joint in obs:
                print(f"  {joint}: {obs[joint]}")
            else:
                print(f"  WARNING: no existe {joint} en observation")

    def validate_pose(self, target_pose: Pose) -> bool:
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

    def print_pose_difference(self, current_pose: Pose, target_pose: Pose):
        print("Diferencias detectadas:")

        for joint in JOINT_KEYS:
            if joint in current_pose and joint in target_pose:
                diff = target_pose[joint] - current_pose[joint]
                print(
                    f"  {joint}: "
                    f"{current_pose[joint]:.3f} -> {target_pose[joint]:.3f} "
                    f"diff={diff:.3f}"
                )

    def smooth_move_to_pose(self, target_pose: Pose, duration=2.0, steps=100):
        """
        Movimiento suave por interpolación lineal.
        Se mantiene como respaldo o comparación contra MPC.
        """

        if not self.connected:
            print("ERROR: El robot no está conectado.")
            return False

        if not self.validate_pose(target_pose):
            return False

        start_pose = self.get_pose()

        print("\n==========================================")
        print("MOVIMIENTO SUAVE A POSE")
        print("==========================================")
        self.print_pose_difference(start_pose, target_pose)

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
        return True

    def mpc_move_to_pose(
        self,
        target_pose: Pose,
        max_iterations=120,
        control_dt=0.03,
        error_tolerance=2.0,
        print_every=10,
        log_label="mpc_motion",
    ):
        """
        Movimiento usando MPC articular con estado comandado interno.

        Registra un CSV con:
        - posiciones reales
        - posiciones comandadas
        - errores por joint
        - error máximo
        - iteración
        - zona o etiqueta del movimiento
        """

        if not self.connected:
            print("ERROR: El robot no está conectado.")
            return False

        if not self.validate_pose(target_pose):
            return False

        current_pose = self.get_pose()
        command_pose = current_pose.copy()

        print("\n==========================================")
        print("MOVIMIENTO MPC A POSE")
        print("MODO: COMMAND-STATE MPC")
        print("==========================================")
        self.print_pose_difference(current_pose, target_pose)

        self.logger.start(label=log_label)

        try:
            for iteration in range(max_iterations):
                current_pose = self.get_pose()

                real_errors = {}

                for joint in JOINT_KEYS:
                    if joint in current_pose and joint in target_pose:
                        real_errors[joint] = target_pose[joint] - current_pose[joint]

                max_real_error = (
                    max(abs(error) for error in real_errors.values())
                    if real_errors
                    else 0.0
                )

                action, _, command_errors, steps = self.mpc.compute_next_action(
                    current_pose=command_pose,
                    target_pose=target_pose,
                )

                previous_command_pose = command_pose.copy()

                for joint in JOINT_KEYS:
                    if joint in action:
                        command_pose[joint] = action[joint]

                self.logger.log_iteration(
                    iteration=iteration,
                    zone=log_label,
                    real_pose=current_pose,
                    command_pose=command_pose,
                    errors=real_errors,
                    max_error=max_real_error,
                )

                if iteration % print_every == 0:
                    print(f"MPC iter={iteration} | max_real_error={max_real_error:.3f}")

                    for joint in JOINT_KEYS:
                        if joint in real_errors and joint in command_pose:
                            delta_cmd = command_pose[joint] - previous_command_pose.get(
                                joint,
                                command_pose[joint],
                            )

                            print(
                                f"  {joint}: "
                                f"real={current_pose[joint]:.3f} | "
                                f"cmd={command_pose[joint]:.3f} | "
                                f"target={target_pose[joint]:.3f} | "
                                f"real_error={real_errors[joint]:.3f} | "
                                f"delta_cmd={delta_cmd:.3f} | "
                                f"step={steps.get(joint, 0.0):.3f}"
                            )

                self.robot.send_action(command_pose)
                time.sleep(control_dt)

                if max_real_error <= error_tolerance:
                    print(
                        f"MPC objetivo alcanzado | "
                        f"iter={iteration} | "
                        f"max_real_error={max_real_error:.3f}"
                    )
                    self.print_current_pose("Pose final leída")
                    return True

            print("MPC terminó por límite de iteraciones.")
            self.print_current_pose("Pose final leída")
            return False

        finally:
            self.logger.close()

    def go_home(self):
        print("\nMoviendo a HOME con MPC...")
        return self.mpc_move_to_pose(
            HOME_POSE,
            max_iterations=120,
            control_dt=0.03,
            error_tolerance=2.0,
            print_every=10,
            log_label="home",
        )

    def mpc_track_continuous(
        self,
        x_norm: float,
        y_norm: float,
        control_dt: float = 0.02,
    ) -> bool:
        """
        Una única iteración MPC de tracking continuo.

        Diseñado para ser llamado repetidamente desde el hilo de control
        del robot mientras el objeto esté visible. No espera convergencia:
        simplemente calcula el próximo paso hacia la pose interpolada
        y lo envía al robot.

        Args:
            x_norm:     Posición X normalizada del objeto [0, 1].
            y_norm:     Posición Y normalizada del objeto [0, 1].
            control_dt: Tiempo de espera después de enviar el comando (segundos).

        Returns:
            True si se envió comando, False si no hay robot conectado o pose.
        """
        if not self.connected:
            return False

        target_pose = self.interpolate_pose_from_position(x_norm, y_norm)

        if target_pose is None:
            return False

        # Inicializar la pose de comando con la pose real la primera vez.
        if self._tracking_command_pose is None:
            self._tracking_command_pose = self.get_pose().copy()

        # Calcular próximo paso con gains de tracking (más agresivo).
        action, max_error, _, _ = self.mpc.compute_next_action_tracking(
            current_pose=self._tracking_command_pose,
            target_pose=target_pose,
        )

        # Actualizar la pose de comando interna.
        for joint in JOINT_KEYS:
            if joint in action:
                self._tracking_command_pose[joint] = action[joint]

        # Enviar al robot.
        self.robot.send_action(self._tracking_command_pose)
        time.sleep(control_dt)

        return True

    def go_ready(self):
        print("\nMoviendo a READY con MPC...")
        return self.mpc_move_to_pose(
            READY_POSE,
            max_iterations=120,
            control_dt=0.03,
            error_tolerance=2.0,
            print_every=10,
            log_label="ready",
        )

    def go_safe_home(self):
        """
        Regresa a una posición segura usando MPC.

        Si SAFE_TRANSITION_POSE existe, primero pasa por ella.
        Si no existe, va directo a SAFE_HOME_POSE.
        """

        print("\n==========================================")
        print("MOVIENDO A SAFE HOME")
        print("CONTROL: COMMAND-STATE MPC")
        print("==========================================")

        if SAFE_TRANSITION_POSE is not None:
            print("Paso 1/2: moviendo a pose de transición segura con MPC...")
            moved_transition = self.mpc_move_to_pose(
                SAFE_TRANSITION_POSE,
                max_iterations=120,
                control_dt=0.03,
                error_tolerance=2.0,
                print_every=10,
                log_label="safe_transition",
            )

            if not moved_transition:
                print("WARNING: No se alcanzó completamente SAFE_TRANSITION_POSE.")

            print("Paso 2/2: moviendo a HOME segura con MPC...")
            moved_home = self.mpc_move_to_pose(
                SAFE_HOME_POSE,
                max_iterations=120,
                control_dt=0.03,
                error_tolerance=2.0,
                print_every=10,
                log_label="safe_home",
            )
        else:
            print("Moviendo directo a HOME segura con MPC...")
            moved_home = self.mpc_move_to_pose(
                SAFE_HOME_POSE,
                max_iterations=120,
                control_dt=0.03,
                error_tolerance=2.0,
                print_every=10,
                log_label="safe_home",
            )

        if moved_home:
            print("Robot en SAFE HOME.")
        else:
            print("WARNING: El robot no llegó completamente a SAFE HOME.")

        return moved_home

    def go_to_zone(self, zone: int):
        if zone not in ZONE_POSES:
            print(f"No existe pose para zona {zone}.")
            return False

        zone_name = ZONE_NAMES.get(zone, "ZONA_DESCONOCIDA")
        target_pose = ZONE_POSES[zone]

        safe_zone_name = (
            str(zone_name)
            .replace(" ", "_")
            .replace("/", "_")
            .replace("Á", "A")
            .replace("É", "E")
            .replace("Í", "I")
            .replace("Ó", "O")
            .replace("Ú", "U")
        )

        log_label = f"zone_{zone}_{safe_zone_name}"

        print("\n==========================================")
        print(f"MOVIENDO A ZONA {zone}: {zone_name}")
        print("CONTROL: COMMAND-STATE MPC")
        print("==========================================")

        print("Valores objetivo importantes:")
        print(f"  shoulder_pan.pos: {target_pose.get('shoulder_pan.pos')}")
        print(f"  shoulder_lift.pos: {target_pose.get('shoulder_lift.pos')}")
        print(f"  elbow_flex.pos: {target_pose.get('elbow_flex.pos')}")
        print(f"  wrist_flex.pos: {target_pose.get('wrist_flex.pos')}")

        return self.mpc_move_to_pose(
            target_pose,
            max_iterations=120,
            control_dt=0.03,
            error_tolerance=2.0,
            print_every=10,
            log_label=log_label,
        )