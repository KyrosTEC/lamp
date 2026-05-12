import cv2


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

    col = min(col, 2)
    row = min(row, 2)

    zone = row * 3 + col + 1

    return zone


def draw_tracking_grid(frame, current_zone=None):
    """
    Dibuja una cuadrícula 3x3 sobre el frame.
    """

    h, w = frame.shape[:2]

    zone_width = w // 3
    zone_height = h // 3

    # Líneas verticales
    cv2.line(frame, (zone_width, 0), (zone_width, h), (255, 255, 255), 2)
    cv2.line(frame, (zone_width * 2, 0), (zone_width * 2, h), (255, 255, 255), 2)

    # Líneas horizontales
    cv2.line(frame, (0, zone_height), (w, zone_height), (255, 255, 255), 2)
    cv2.line(frame, (0, zone_height * 2), (w, zone_height * 2), (255, 255, 255), 2)

    zone_number = 1

    for row in range(3):
        for col in range(3):
            x = col * zone_width + 20
            y = row * zone_height + 40

            color = (0, 255, 255)

            if current_zone == zone_number:
                color = (0, 0, 255)

            cv2.putText(
                frame,
                str(zone_number),
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                color,
                3
            )

            zone_number += 1

    return frame