import numpy as np


JOINT_KEYS = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]


class MPCJointController:
    """
    MPC articular simplificado para el SO-101.

    Modelo:
        q(k+1) = q(k) + u(k)

    Donde:
        q = posiciones articulares
        u = incremento de posición articular

    Costo conceptual:
        J = sum( ||q - q_target||_Q^2 + ||u||_R^2 )

    Esta implementación calcula pasos incrementales limitados por articulación.
    El estado comandado acumulado se maneja desde SO101Controller.
    """

    def __init__(
        self,
        horizon=12,
        max_step_deg=3.0,
        position_tolerance=2.0,
        control_gain=0.65,
    ):
        self.horizon = horizon
        self.max_step_deg = max_step_deg
        self.position_tolerance = position_tolerance
        self.control_gain = control_gain

        # Peso del error de posición.
        # Más alto = corrige con más prioridad ese joint.
        self.q_weights = {
            "shoulder_pan.pos": 1.2,
            "shoulder_lift.pos": 1.3,
            "elbow_flex.pos": 1.3,
            "wrist_flex.pos": 1.0,
            "wrist_roll.pos": 0.8,
            "gripper.pos": 0.4,
        }

        # Peso del esfuerzo de control.
        # Más alto = movimiento más conservador.
        self.r_weights = {
            "shoulder_pan.pos": 0.7,
            "shoulder_lift.pos": 0.6,
            "elbow_flex.pos": 0.6,
            "wrist_flex.pos": 0.7,
            "wrist_roll.pos": 0.7,
            "gripper.pos": 0.5,
        }

        # Máximo cambio permitido por iteración.
        # Ajustado para demo: más rápido, pero todavía controlado.
        self.max_step_by_joint = {
            "shoulder_pan.pos": 2.5,
            "shoulder_lift.pos": 4.0,
            "elbow_flex.pos": 4.0,
            "wrist_flex.pos": 3.0,
            "wrist_roll.pos": 2.5,
            "gripper.pos": 3.0,
        }

        # Gains más agresivos para modo tracking continuo.
        # Mayor control_gain + mayor max_step para joints críticos.
        self.tracking_control_gain = 0.80
        self.tracking_max_step_by_joint = {
            "shoulder_pan.pos": 4.0,
            "shoulder_lift.pos": 5.5,
            "elbow_flex.pos": 5.0,
            "wrist_flex.pos": 4.0,
            "wrist_roll.pos": 3.0,
            "gripper.pos": 3.0,
        }
        # R weights reducidos para modo tracking (más reactivo)
        self.tracking_r_weights = {
            "shoulder_pan.pos": 0.4,
            "shoulder_lift.pos": 0.35,
            "elbow_flex.pos": 0.4,
            "wrist_flex.pos": 0.5,
            "wrist_roll.pos": 0.5,
            "gripper.pos": 0.5,
        }

    def compute_joint_error(self, current_pose, target_pose):
        errors = {}

        for joint in JOINT_KEYS:
            if joint in current_pose and joint in target_pose:
                errors[joint] = float(target_pose[joint] - current_pose[joint])

        return errors

    def max_abs_error(self, current_pose, target_pose):
        errors = self.compute_joint_error(current_pose, target_pose)

        if not errors:
            return 0.0

        return max(abs(error) for error in errors.values())

    def is_target_reached(self, current_pose, target_pose):
        return self.max_abs_error(current_pose, target_pose) <= self.position_tolerance

    def compute_next_action(self, current_pose, target_pose):
        """
        Calcula la siguiente acción incremental.

        Args:
            current_pose:
                Puede ser la pose real o una pose comandada interna.
            target_pose:
                Pose objetivo.

        Returns:
            action:
                Diccionario compatible con robot.send_action().
            max_error:
                Error máximo absoluto.
            errors:
                Error por joint.
            steps:
                Paso calculado por joint.
        """

        action = current_pose.copy()
        errors = self.compute_joint_error(current_pose, target_pose)
        steps = {}

        for joint in JOINT_KEYS:
            if joint not in errors:
                continue

            current_value = float(current_pose[joint])
            error = errors[joint]

            q_weight = self.q_weights.get(joint, 1.0)
            r_weight = self.r_weights.get(joint, 1.0)

            effective_gain = self.control_gain * (q_weight / (q_weight + r_weight))
            desired_step = effective_gain * error

            max_step = self.max_step_by_joint.get(joint, self.max_step_deg)
            step = float(np.clip(desired_step, -max_step, max_step))

            action[joint] = current_value + step
            steps[joint] = step

        max_error = max(abs(error) for error in errors.values()) if errors else 0.0

        return action, max_error, errors, steps

    def compute_next_action_tracking(self, current_pose, target_pose):
        """
        Variante de compute_next_action con gains más agresivos para
        tracking continuo. Usa tracking_control_gain y tracking_max_step_by_joint.

        Diseñado para llamarse desde un bucle externo que actualiza el target
        en cada iteración (no espera convergencia completa).

        Returns:
            action, max_error, errors, steps  (mismo formato que compute_next_action)
        """
        action = current_pose.copy()
        errors = self.compute_joint_error(current_pose, target_pose)
        steps = {}

        for joint in JOINT_KEYS:
            if joint not in errors:
                continue

            current_value = float(current_pose[joint])
            error = errors[joint]

            q_weight = self.q_weights.get(joint, 1.0)
            r_weight = self.tracking_r_weights.get(joint, 0.5)

            effective_gain = self.tracking_control_gain * (q_weight / (q_weight + r_weight))
            desired_step = effective_gain * error

            max_step = self.tracking_max_step_by_joint.get(joint, self.max_step_deg)
            step = float(np.clip(desired_step, -max_step, max_step))

            action[joint] = current_value + step
            steps[joint] = step

        max_error = max(abs(error) for error in errors.values()) if errors else 0.0

        return action, max_error, errors, steps