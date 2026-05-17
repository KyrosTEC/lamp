import cv2
import numpy as np


CAMERA_INDEX = 0


def detect_position_only(frame):
    """
    Versión ligera del detector: solo retorna datos de posición, sin dibujar.

    Más eficiente cuando el resultado se usa solo para control (no display).

    Returns:
        robot_target: dict con posición o None.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_green = np.array([35, 80, 80])
    upper_green = np.array([90, 255, 255])

    mask = cv2.inRange(hsv, lower_green, upper_green)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)

    if area <= 800:
        return None

    x, y, w, h = cv2.boundingRect(largest_contour)
    center_x = x + w // 2
    center_y = y + h // 2

    frame_h, frame_w = frame.shape[:2]

    return {
        "x": center_x,
        "y": center_y,
        "x_norm": center_x / frame_w,
        "y_norm": center_y / frame_h,
        "area": area,
        "bbox": (x, y, w, h),
    }


def detect_neon_green_paper(frame):
    """
    Detecta un objeto color verde fosforescente tipo #39FF14.
    Regresa:
      result: frame con dibujos
      debug_mask: máscara para depuración
      robot_target: dict con posición normalizada o None
    """

    result = frame.copy()

    # Convertir BGR a HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Rango HSV aproximado para verde fosforescente #39FF14
    # Puedes ajustar estos valores si tu iluminación cambia.
    lower_green = np.array([35, 80, 80])
    upper_green = np.array([90, 255, 255])

    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Limpiar ruido
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Encontrar contornos
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    robot_target = None

    if contours:
        # Tomar el contorno más grande
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)

        # Área mínima para evitar falsos positivos
        if area > 800:
            x, y, w, h = cv2.boundingRect(largest_contour)

            center_x = x + w // 2
            center_y = y + h // 2

            frame_h, frame_w = frame.shape[:2]

            x_norm = center_x / frame_w
            y_norm = center_y / frame_h

            robot_target = {
                "x": center_x,
                "y": center_y,
                "x_norm": x_norm,
                "y_norm": y_norm,
                "area": area,
                "bbox": (x, y, w, h),
            }

            # Dibujar bounding box
            cv2.rectangle(
                result,
                (x, y),
                (x + w, y + h),
                (57, 255, 20),
                3
            )

            # Dibujar centro
            cv2.circle(
                result,
                (center_x, center_y),
                8,
                (0, 0, 255),
                -1
            )

            # Texto de detección
            cv2.putText(
                result,
                f"POSTIT DETECTADO | area={int(area)}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (57, 255, 20),
                2
            )

    return result, mask, robot_target