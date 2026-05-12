import cv2
import time

from vision_neon_green import detect_neon_green_paper, CAMERA_INDEX
from so101_controller import SO101Controller


USE_ROBOT = True
DETECT_EVERY_N_FRAMES = 2

FRAMES_TO_CONFIRM_DETECTED = 8
FRAMES_TO_CONFIRM_NOT_DETECTED = 10

PRINT_DETECTION_EVERY_N_FRAMES = 15


def get_tracking_zone(x, y, frame_width, frame_height):
    """
    Divide la imagen en 9 zonas y regresa el número de zona.

    Zonas:
        1 | 2 | 3
        4 | 5 | 6
        7 | 8 | 9
    """

    zone_width = frame_width / 3
    zone_height = frame_height / 3

    col = int(x // zone_width)
    row = int(y // zone_height)

    col = max(0, min(col, 2))
    row = max(0, min(row, 2))

    zone = row * 3 + col + 1

    return zone


def get_zone_name(zone):
    zone_names = {
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

    return zone_names.get(zone, "SIN ZONA")


def draw_tracking_grid(frame, current_zone=None):
    h, w = frame.shape[:2]

    zone_width = w // 3
    zone_height = h // 3

    cv2.line(frame, (zone_width, 0), (zone_width, h), (255, 255, 255), 2)
    cv2.line(frame, (zone_width * 2, 0), (zone_width * 2, h), (255, 255, 255), 2)

    cv2.line(frame, (0, zone_height), (w, zone_height), (255, 255, 255), 2)
    cv2.line(frame, (0, zone_height * 2), (w, zone_height * 2), (255, 255, 255), 2)

    zone_number = 1

    for row in range(3):
        for col in range(3):
            x = col * zone_width + 20
            y = row * zone_height + 45

            if current_zone == zone_number:
                color = (0, 0, 255)
                thickness = 4
            else:
                color = (0, 255, 255)
                thickness = 2

            cv2.putText(
                frame,
                str(zone_number),
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                color,
                thickness,
            )

            zone_number += 1

    return frame


def draw_zone_ranges(frame):
    h, w = frame.shape[:2]

    zone_width = w // 3
    zone_height = h // 3

    text = f"Frame: {w}x{h} | Zona aprox: {zone_width}x{zone_height}"

    cv2.putText(
        frame,
        text,
        (30, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
    )

    return frame


def move_robot_to_zone(robot, zone):
    if robot is None:
        return False

    if not hasattr(robot, "go_to_zone"):
        print("ERROR: Tu SO101Controller no tiene el método go_to_zone(zone).")
        return False

    try:
        print(f"Robot apuntando a zona {zone}: {get_zone_name(zone)}")
        robot.go_to_zone(zone)
        return True
    except Exception as e:
        print(f"ERROR al mover robot a zona {zone}: {e}")
        return False


def move_robot_to_safe_home(robot):
    if robot is None:
        return False

    try:
        if hasattr(robot, "go_safe_home"):
            robot.go_safe_home()
        else:
            robot.go_home()

        return True
    except Exception as e:
        print(f"ERROR al mover robot a SAFE HOME: {e}")
        return False


def reset_tracking_state():
    return {
        "current_zone": None,
        "confirmed_zone": None,
        "last_commanded_zone": None,
        "last_robot_target": None,
        "last_seen_zone": None,
    }


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print("No se pudo abrir la camara.")
        return

    robot = None
    robot_connected = False

    frame_count = 0
    detected_counter = 0
    not_detected_counter = 0

    tracking = reset_tracking_state()

    current_state = "CAMERA_ONLY"
    auto_mode = False
    already_safe_homed = False

    print("Sistema iniciado.")
    print("Detectando color verde fosforescente tipo #39FF14.")
    print("Dividiendo la imagen en 9 zonas.")
    print("Teclas:")
    print("  r = conectar/calibrar robot")
    print("  h = mover a SAFE HOME manual")
    print("  g = mover a READY manual")
    print("  a = activar/desactivar modo automatico")
    print("  q = salir")

    try:
        for _ in range(10):
            cap.read()

        while True:
            ret, frame = cap.read()

            if not ret:
                print("No se pudo leer frame.")
                break

            frame_count += 1

            result = frame.copy()
            debug_mask = None
            robot_target = None

            if frame_count % DETECT_EVERY_N_FRAMES == 0:
                result, debug_mask, robot_target = detect_neon_green_paper(frame)
                color_detected = robot_target is not None

                if color_detected:
                    detected_counter += 1
                    not_detected_counter = 0

                    frame_h, frame_w = frame.shape[:2]

                    current_zone = get_tracking_zone(
                        robot_target["x"],
                        robot_target["y"],
                        frame_w,
                        frame_h,
                    )

                    tracking["current_zone"] = current_zone
                    tracking["last_robot_target"] = robot_target
                    tracking["last_seen_zone"] = current_zone

                    if frame_count % PRINT_DETECTION_EVERY_N_FRAMES == 0:
                        print(
                            f"Objeto detectado | "
                            f"x={robot_target['x']} | "
                            f"y={robot_target['y']} | "
                            f"zona={current_zone} | "
                            f"{get_zone_name(current_zone)}"
                        )

                else:
                    not_detected_counter += 1
                    detected_counter = 0
                    tracking["current_zone"] = None
                    tracking["confirmed_zone"] = None
                    tracking["last_robot_target"] = None
                    tracking["last_seen_zone"] = None

                if (
                    color_detected
                    and detected_counter >= FRAMES_TO_CONFIRM_DETECTED
                    and tracking["current_zone"] is not None
                ):
                    tracking["confirmed_zone"] = tracking["current_zone"]

                if (
                    auto_mode
                    and robot_connected
                    and tracking["confirmed_zone"] is not None
                    and tracking["confirmed_zone"] != tracking["last_commanded_zone"]
                ):
                    confirmed_zone = tracking["confirmed_zone"]

                    print(
                        f"Zona confirmada: {confirmed_zone} - "
                        f"{get_zone_name(confirmed_zone)}. Apuntando robot."
                    )

                    moved = move_robot_to_zone(robot, confirmed_zone)

                    if moved:
                        current_state = f"ZONE_{confirmed_zone}"
                        tracking["last_commanded_zone"] = confirmed_zone
                        already_safe_homed = False

                if (
                    auto_mode
                    and robot_connected
                    and not_detected_counter >= FRAMES_TO_CONFIRM_NOT_DETECTED
                    and not already_safe_homed
                ):
                    print("Post-it no detectado. Moviendo a SAFE HOME.")

                    moved = move_robot_to_safe_home(robot)

                    if moved:
                        current_state = "SAFE_HOME"
                        already_safe_homed = True

                    tracking = reset_tracking_state()

            else:
                tracking["current_zone"] = tracking["last_seen_zone"]

            result = draw_tracking_grid(result, tracking["current_zone"])
            result = draw_zone_ranges(result)

            if tracking["last_robot_target"] is not None and tracking["last_seen_zone"] is not None:
                text = (
                    f"STATE: {current_state} | AUTO: {auto_mode} | "
                    f"x={tracking['last_robot_target']['x']}, "
                    f"y={tracking['last_robot_target']['y']} | "
                    f"zone={tracking['last_seen_zone']}"
                )
            else:
                text = f"STATE: {current_state} | AUTO: {auto_mode} | zone=None"

            cv2.putText(
                result,
                text,
                (30, 210),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
            )

            if tracking["confirmed_zone"] is not None:
                confirmed_text = (
                    f"CONFIRMED ZONE: {tracking['confirmed_zone']} - "
                    f"{get_zone_name(tracking['confirmed_zone'])}"
                )
            else:
                confirmed_text = "CONFIRMED ZONE: None"

            cv2.putText(
                result,
                confirmed_text,
                (30, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
            )

            cv2.putText(
                result,
                f"det={detected_counter}/{FRAMES_TO_CONFIRM_DETECTED} | "
                f"no_det={not_detected_counter}/{FRAMES_TO_CONFIRM_NOT_DETECTED}",
                (30, 270),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
            )

            cv2.putText(
                result,
                "r=robot | h=SAFE_HOME | g=READY | a=AUTO | q=salir",
                (30, 300),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
            )

            cv2.imshow("Deteccion post-it verde + zonas", result)

            if debug_mask is not None:
                cv2.imshow("Debug mascara verde", debug_mask)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                print("Saliendo del programa...")

                if robot_connected and robot is not None and not already_safe_homed:
                    print("Regresando a SAFE HOME antes de salir...")
                    if move_robot_to_safe_home(robot):
                        already_safe_homed = True
                        current_state = "SAFE_HOME"

                break

            if key == ord("r") and USE_ROBOT and not robot_connected:
                print("Conectando/calibrando SO-101...")

                try:
                    robot = SO101Controller()
                    robot.connect()
                    robot_connected = True
                    current_state = "ROBOT_CONNECTED"
                    already_safe_homed = False
                    print("SO-101 listo.")
                except Exception as e:
                    print(f"ERROR al conectar SO-101: {e}")
                    robot = None
                    robot_connected = False

            if key == ord("h") and USE_ROBOT and robot_connected:
                print("Moviendo manualmente a SAFE HOME...")

                auto_mode = False

                if move_robot_to_safe_home(robot):
                    current_state = "SAFE_HOME"
                    already_safe_homed = True

                tracking = reset_tracking_state()
                detected_counter = 0
                not_detected_counter = 0

            if key == ord("g") and USE_ROBOT and robot_connected:
                print("Moviendo manualmente a READY...")

                auto_mode = False

                try:
                    robot.go_ready()
                    current_state = "READY"
                    already_safe_homed = False
                except Exception as e:
                    print(f"ERROR al mover a READY: {e}")

                tracking = reset_tracking_state()
                detected_counter = 0
                not_detected_counter = 0

            if key == ord("a"):
                if not robot_connected:
                    print("Primero conecta el robot con r.")
                else:
                    auto_mode = not auto_mode

                    detected_counter = 0
                    not_detected_counter = 0
                    tracking = reset_tracking_state()

                    print(f"Modo automatico: {auto_mode}")

    finally:
        cap.release()
        cv2.destroyAllWindows()

        if robot_connected and robot is not None:
            if not already_safe_homed:
                try:
                    print("Asegurando SAFE HOME antes de desconectar...")
                    move_robot_to_safe_home(robot)
                except Exception as e:
                    print(f"No se pudo mover a SAFE HOME antes de desconectar: {e}")

            robot.disconnect()


if __name__ == "__main__":
    main()