import cv2
import threading
import glob
import os
import subprocess
import sys
import time

try:
    import serial
except ImportError:
    serial = None

from vision_neon_green import detect_neon_green_paper, CAMERA_INDEX
from so101_controller import SO101Controller


USE_ROBOT = True
USE_LEDS = True

ESP32_LED_PORT = "/dev/ttyUSB0"
ESP32_LED_BAUDRATE = 115200

DETECT_EVERY_N_FRAMES = 2

FRAMES_TO_CONFIRM_DETECTED = 8
FRAMES_TO_CONFIRM_NOT_DETECTED = 10

PRINT_DETECTION_EVERY_N_FRAMES = 15

LOGS_DIR = "logs"
PLOTS_DIR = "plots"
OPEN_PLOTS_ON_EXIT = True


class LEDController:
    """
    Controla los LEDs conectados a la ESP32 de forma automática.

    Python manda:
    p = prender LEDs
    o = apagar LEDs
    """

    def __init__(self, port="/dev/ttyUSB0", baudrate=115200, enabled=True):
        self.port = port
        self.baudrate = baudrate
        self.enabled = enabled
        self.esp32 = None
        self.current_state = None

    def connect(self):
        if not self.enabled:
            print("[LEDController] Control de LEDs desactivado.")
            return

        if serial is None:
            print("[LEDController] pyserial no está instalado. Ejecuta: pip install pyserial")
            self.enabled = False
            return

        try:
            self.esp32 = serial.Serial(self.port, self.baudrate, timeout=1)
            time.sleep(2)
            self.off()
            print(f"[LEDController] ESP32 conectada en {self.port}")
        except Exception as e:
            self.esp32 = None
            self.enabled = False
            print(f"[LEDController] No se pudo conectar la ESP32 en {self.port}: {e}")
            print("[LEDController] El programa continuará sin LEDs.")

    def send(self, command):
        if not self.enabled:
            return

        if self.esp32 is None:
            return

        if not self.esp32.is_open:
            return

        try:
            self.esp32.write(command.encode("utf-8"))
        except Exception as e:
            print(f"[LEDController] Error enviando comando a ESP32: {e}")

    def on(self):
        if self.current_state != "on":
            self.send("p")
            self.current_state = "on"
            print("[LEDController] LEDs ON")

    def off(self):
        if self.current_state != "off":
            self.send("o")
            self.current_state = "off"
            print("[LEDController] LEDs OFF")

    def close(self):
        self.off()

        if self.esp32 is not None and self.esp32.is_open:
            self.esp32.close()
            print("[LEDController] ESP32 desconectada")


def get_tracking_zone(x, y, frame_width, frame_height):
    zone_width = frame_width / 3
    zone_height = frame_height / 3

    col = int(x // zone_width)
    row = int(y // zone_height)

    col = max(0, min(col, 2))
    row = max(0, min(row, 2))

    return row * 3 + col + 1


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
            x_text = col * zone_width + 20
            y_text = row * zone_height + 45

            if current_zone == zone_number:
                color = (0, 0, 255)
                thickness = 4
            else:
                color = (0, 255, 255)
                thickness = 2

            cv2.putText(
                frame,
                str(zone_number),
                (x_text, y_text),
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


def reset_tracking_state():
    return {
        "current_zone": None,
        "confirmed_zone": None,
        "last_commanded_zone": None,
        "last_robot_target": None,
        "last_seen_zone": None,
    }


def create_robot_task_state():
    return {
        "busy": False,
        "completed": False,
        "success": False,
        "error": None,
        "label": None,
        "target_state": None,
        "thread": None,
    }


def is_robot_busy(robot_task, robot_task_lock):
    with robot_task_lock:
        return robot_task["busy"]


def start_robot_task(
    robot_task,
    robot_task_lock,
    label,
    target_state,
    task_function,
):
    """
    Ejecuta una acción del robot en segundo plano.
    Esto evita que OpenCV se congele mientras el MPC mueve el robot.
    """

    with robot_task_lock:
        if robot_task["busy"]:
            print(f"Robot ocupado. No se inicia nueva tarea: {label}")
            return False

        robot_task["busy"] = True
        robot_task["completed"] = False
        robot_task["success"] = False
        robot_task["error"] = None
        robot_task["label"] = label
        robot_task["target_state"] = target_state

    def worker():
        success = False
        error = None

        try:
            print(f"Iniciando tarea robot: {label}")
            result = task_function()
            success = bool(result) if result is not None else True
            print(f"Tarea robot terminada: {label}")
        except Exception as e:
            error = str(e)
            print(f"ERROR en tarea robot {label}: {e}")

        with robot_task_lock:
            robot_task["busy"] = False
            robot_task["completed"] = True
            robot_task["success"] = success
            robot_task["error"] = error

    thread = threading.Thread(target=worker, daemon=True)

    with robot_task_lock:
        robot_task["thread"] = thread

    thread.start()
    return True


def consume_robot_task_result(robot_task, robot_task_lock):
    """
    Lee el resultado de la última tarea terminada.
    """

    with robot_task_lock:
        if not robot_task["completed"]:
            return None

        result = {
            "success": robot_task["success"],
            "error": robot_task["error"],
            "label": robot_task["label"],
            "target_state": robot_task["target_state"],
        }

        robot_task["completed"] = False
        return result


def wait_for_robot_task(robot_task, robot_task_lock):
    """
    Espera a que termine una tarea del robot antes de desconectar.
    """

    with robot_task_lock:
        thread = robot_task.get("thread")

    if thread is not None and thread.is_alive():
        print("Esperando a que termine la tarea actual del robot...")
        thread.join()


def move_robot_to_zone(robot, zone):
    if robot is None:
        return False

    if not hasattr(robot, "go_to_zone"):
        print("ERROR: SO101Controller no tiene el método go_to_zone(zone).")
        return False

    print(f"Robot apuntando a zona {zone}: {get_zone_name(zone)}")
    return bool(robot.go_to_zone(zone))


def move_robot_to_safe_home(robot):
    if robot is None:
        return False

    if hasattr(robot, "go_safe_home"):
        return bool(robot.go_safe_home())

    return bool(robot.go_home())


def get_session_csv_logs(session_start_time, logs_dir=LOGS_DIR):
    """
    Obtiene los CSV generados desde que inició esta ejecución del programa.
    """

    csv_paths = glob.glob(os.path.join(logs_dir, "*.csv"))
    session_csvs = []

    for csv_path in csv_paths:
        try:
            modified_time = os.path.getmtime(csv_path)

            if modified_time >= session_start_time:
                session_csvs.append(csv_path)

        except OSError:
            continue

    session_csvs.sort(key=os.path.getmtime)

    return session_csvs


def generate_plots_for_session(session_start_time):
    """
    Genera automáticamente las gráficas para todos los CSV creados durante
    esta ejecución del programa.
    """

    csv_logs = get_session_csv_logs(session_start_time)

    if not csv_logs:
        print("No se encontraron CSV nuevos para generar gráficas.")
        return

    if not os.path.exists("tools/plot_mpc_log.py"):
        print("No se encontró tools/plot_mpc_log.py. No se generaron gráficas.")
        return

    print("\n==========================================")
    print("GENERANDO GRÁFICAS MPC")
    print("==========================================")

    generated_any = False

    for csv_path in csv_logs:
        print(f"Generando gráficas para: {csv_path}")

        result = subprocess.run(
            [
                sys.executable,
                "tools/plot_mpc_log.py",
                csv_path,
                "--output-dir",
                PLOTS_DIR,
            ],
            text=True,
        )

        if result.returncode != 0:
            print(f"WARNING: No se pudieron generar gráficas para {csv_path}")
        else:
            generated_any = True

    if generated_any:
        print("\nGráficas generadas en:")
        print(PLOTS_DIR)

        if OPEN_PLOTS_ON_EXIT:
            try:
                subprocess.Popen(["xdg-open", PLOTS_DIR])
            except Exception as e:
                print(f"No se pudo abrir la carpeta de gráficas automáticamente: {e}")
    else:
        print("No se generó ninguna gráfica.")


def draw_ui(
    frame,
    current_state,
    auto_mode,
    tracking,
    detected_counter,
    not_detected_counter,
    robot_busy,
    led_state,
):
    result = draw_tracking_grid(frame, tracking["current_zone"])
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

    robot_status = "BUSY" if robot_busy else "IDLE"

    cv2.putText(
        result,
        f"CONTROL: MPC ASYNC | ROBOT: {robot_status} | LEDS: {led_state}",
        (30, 330),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2,
    )

    return result


def update_leds_for_state(
    leds,
    auto_mode,
    robot_connected,
    current_state,
    detected_counter,
    tracking,
):
    """
    Reglas:
    - Si está en SAFE_HOME, LEDs apagados.
    - Si no está en modo automático, LEDs apagados.
    - Si no hay robot conectado, LEDs apagados.
    - Si está detectando objetivo verde en automático, LEDs encendidos.
    - Si no detecta objetivo, LEDs apagados.
    """

    if current_state == "SAFE_HOME":
        leds.off()
        return "OFF"

    if not auto_mode:
        leds.off()
        return "OFF"

    if not robot_connected:
        leds.off()
        return "OFF"

    target_visible = (
        detected_counter > 0
        and tracking["current_zone"] is not None
        and tracking["last_robot_target"] is not None
    )

    if target_visible:
        leds.on()
        return "ON"

    leds.off()
    return "OFF"


def main():
    session_start_time = time.time()

    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print("No se pudo abrir la camara.")
        return

    leds = LEDController(
        port=ESP32_LED_PORT,
        baudrate=ESP32_LED_BAUDRATE,
        enabled=USE_LEDS,
    )
    leds.connect()

    robot = None
    robot_connected = False

    robot_task = create_robot_task_state()
    robot_task_lock = threading.Lock()

    frame_count = 0
    detected_counter = 0
    not_detected_counter = 0

    tracking = reset_tracking_state()

    current_state = "CAMERA_ONLY"
    auto_mode = False
    already_safe_homed = False
    led_state = "OFF"

    print("Sistema iniciado.")
    print("Detectando color verde fosforescente tipo #39FF14.")
    print("Dividiendo la imagen en 9 zonas.")
    print("Control activo: MPC articular en segundo plano.")
    print("Control LEDs: automatico por ESP32.")
    print("Al salir, se generarán automáticamente las gráficas MPC.")
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
            task_result = consume_robot_task_result(robot_task, robot_task_lock)

            if task_result is not None:
                if task_result["success"]:
                    current_state = task_result["target_state"]

                    if current_state == "SAFE_HOME":
                        already_safe_homed = True
                        leds.off()
                        led_state = "OFF"
                    else:
                        already_safe_homed = False

                    print(f"Tarea completada correctamente: {task_result['label']}")
                else:
                    print(
                        f"Tarea fallida: {task_result['label']} | "
                        f"error={task_result['error']}"
                    )

            ret, frame = cap.read()

            if not ret:
                print("No se pudo leer frame.")
                leds.off()
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

                led_state = update_leds_for_state(
                    leds=leds,
                    auto_mode=auto_mode,
                    robot_connected=robot_connected,
                    current_state=current_state,
                    detected_counter=detected_counter,
                    tracking=tracking,
                )

                if (
                    color_detected
                    and detected_counter >= FRAMES_TO_CONFIRM_DETECTED
                    and tracking["current_zone"] is not None
                ):
                    tracking["confirmed_zone"] = tracking["current_zone"]

                robot_busy = is_robot_busy(robot_task, robot_task_lock)

                if (
                    auto_mode
                    and robot_connected
                    and not robot_busy
                    and tracking["confirmed_zone"] is not None
                    and tracking["confirmed_zone"] != tracking["last_commanded_zone"]
                ):
                    confirmed_zone = tracking["confirmed_zone"]

                    print(
                        f"Zona confirmada: {confirmed_zone} - "
                        f"{get_zone_name(confirmed_zone)}. Apuntando robot con MPC."
                    )

                    started = start_robot_task(
                        robot_task=robot_task,
                        robot_task_lock=robot_task_lock,
                        label=f"go_to_zone_{confirmed_zone}",
                        target_state=f"ZONE_{confirmed_zone}",
                        task_function=lambda zone=confirmed_zone: move_robot_to_zone(robot, zone),
                    )

                    if started:
                        tracking["last_commanded_zone"] = confirmed_zone
                        already_safe_homed = False

                robot_busy = is_robot_busy(robot_task, robot_task_lock)

                if (
                    auto_mode
                    and robot_connected
                    and not robot_busy
                    and not_detected_counter >= FRAMES_TO_CONFIRM_NOT_DETECTED
                    and not already_safe_homed
                ):
                    print("Post-it no detectado. Moviendo a SAFE HOME con MPC.")

                    leds.off()
                    led_state = "OFF"

                    started = start_robot_task(
                        robot_task=robot_task,
                        robot_task_lock=robot_task_lock,
                        label="go_safe_home",
                        target_state="SAFE_HOME",
                        task_function=lambda: move_robot_to_safe_home(robot),
                    )

                    if started:
                        tracking = reset_tracking_state()

            else:
                tracking["current_zone"] = tracking["last_seen_zone"]

            robot_busy = is_robot_busy(robot_task, robot_task_lock)

            result = draw_ui(
                result,
                current_state=current_state,
                auto_mode=auto_mode,
                tracking=tracking,
                detected_counter=detected_counter,
                not_detected_counter=not_detected_counter,
                robot_busy=robot_busy,
                led_state=led_state,
            )

            cv2.imshow("Deteccion post-it verde + zonas + MPC", result)

            if debug_mask is not None:
                cv2.imshow("Debug mascara verde", debug_mask)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                print("Saliendo del programa...")

                leds.off()
                led_state = "OFF"

                wait_for_robot_task(robot_task, robot_task_lock)

                if robot_connected and robot is not None and not already_safe_homed:
                    print("Regresando a SAFE HOME antes de salir con MPC...")
                    move_robot_to_safe_home(robot)
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
                    leds.off()
                    led_state = "OFF"
                    print("SO-101 listo.")
                except Exception as e:
                    print(f"ERROR al conectar SO-101: {e}")
                    robot = None
                    robot_connected = False
                    leds.off()
                    led_state = "OFF"

            if key == ord("h") and USE_ROBOT and robot_connected:
                print("Moviendo manualmente a SAFE HOME con MPC...")

                auto_mode = False
                leds.off()
                led_state = "OFF"

                wait_for_robot_task(robot_task, robot_task_lock)

                started = start_robot_task(
                    robot_task=robot_task,
                    robot_task_lock=robot_task_lock,
                    label="manual_safe_home",
                    target_state="SAFE_HOME",
                    task_function=lambda: move_robot_to_safe_home(robot),
                )

                if started:
                    tracking = reset_tracking_state()
                    detected_counter = 0
                    not_detected_counter = 0

            if key == ord("g") and USE_ROBOT and robot_connected:
                print("Moviendo manualmente a READY con MPC...")

                auto_mode = False
                leds.off()
                led_state = "OFF"

                wait_for_robot_task(robot_task, robot_task_lock)

                started = start_robot_task(
                    robot_task=robot_task,
                    robot_task_lock=robot_task_lock,
                    label="manual_ready",
                    target_state="READY",
                    task_function=lambda: robot.go_ready(),
                )

                if started:
                    tracking = reset_tracking_state()
                    detected_counter = 0
                    not_detected_counter = 0
                    already_safe_homed = False

            if key == ord("a"):
                if not robot_connected:
                    print("Primero conecta el robot con r.")
                    leds.off()
                    led_state = "OFF"
                else:
                    auto_mode = not auto_mode

                    detected_counter = 0
                    not_detected_counter = 0
                    tracking = reset_tracking_state()

                    if not auto_mode:
                        leds.off()
                        led_state = "OFF"

                    print(f"Modo automatico: {auto_mode}")

    finally:
        leds.off()

        cap.release()
        cv2.destroyAllWindows()

        wait_for_robot_task(robot_task, robot_task_lock)

        if robot_connected and robot is not None:
            if not already_safe_homed:
                try:
                    print("Asegurando SAFE HOME antes de desconectar con MPC...")
                    leds.off()
                    move_robot_to_safe_home(robot)
                except Exception as e:
                    print(f"No se pudo mover a SAFE HOME antes de desconectar: {e}")

            robot.disconnect()

        leds.close()

        generate_plots_for_session(session_start_time)


if __name__ == "__main__":
    main()