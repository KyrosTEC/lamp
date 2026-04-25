import cv2
import time

from vision_book import detect_open_book, CAMERA_INDEX
from so101_controller import SO101Controller


DETECT_EVERY_N_FRAMES = 2

# Para evitar que el brazo se mueva por falsos positivos/negativos
FRAMES_TO_CONFIRM_DETECTED = 5
FRAMES_TO_CONFIRM_NOT_DETECTED = 8


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print("No se pudo abrir la camara.")
        return

    robot = SO101Controller()

    frame_count = 0
    detected_counter = 0
    not_detected_counter = 0

    current_state = "UNKNOWN"

    try:
        robot.connect()

        print("Mandando brazo a HOME inicial...")
        robot.go_home()
        current_state = "HOME"

        time.sleep(1)

        for _ in range(10):
            cap.read()

        print("Sistema iniciado.")
        print("Si detecta libro -> READY")
        print("Si no detecta libro -> HOME")
        print("Presiona q para salir.")

        while True:
            ret, frame = cap.read()

            if not ret:
                print("No se pudo leer frame.")
                break

            frame_count += 1

            if frame_count % DETECT_EVERY_N_FRAMES != 0:
                cv2.imshow("Deteccion libro abierto", frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break

                continue

            result, debug_edges, robot_target = detect_open_book(frame)

            book_detected = robot_target is not None

            if book_detected:
                detected_counter += 1
                not_detected_counter = 0
            else:
                not_detected_counter += 1
                detected_counter = 0

            if detected_counter >= FRAMES_TO_CONFIRM_DETECTED and current_state != "READY":
                print("Libro confirmado. Cambiando a READY.")
                robot.go_ready()
                current_state = "READY"

            if not_detected_counter >= FRAMES_TO_CONFIRM_NOT_DETECTED and current_state != "HOME":
                print("Libro no detectado. Regresando a HOME.")
                robot.go_home()
                current_state = "HOME"

            if robot_target is not None:
                cv2.putText(
                    result,
                    f"STATE: {current_state} | x_norm={robot_target['x_norm']:.3f}, y_norm={robot_target['y_norm']:.3f}",
                    (30, 210),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 255),
                    2
                )
            else:
                cv2.putText(
                    result,
                    f"STATE: {current_state}",
                    (30, 210),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 255),
                    2
                )

            cv2.imshow("Deteccion libro abierto", result)
            cv2.imshow("Debug bordes", debug_edges)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        robot.disconnect()


if __name__ == "__main__":
    main()