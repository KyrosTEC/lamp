import cv2
import numpy as np

CAMERA_INDEX = 0

MIN_AREA = 12000
CANNY_LOW = 50
CANNY_HIGH = 150


def detect_open_book(frame):
    output = frame.copy()
    h, w = frame.shape[:2]

    # ROI: por ahora toda la imagen.
    # Luego puedes limitarla a la zona exacta de la mesa.
    roi = frame.copy()

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blur, CANNY_LOW, CANNY_HIGH)

    kernel = np.ones((5, 5), np.uint8)
    edges_clean = cv2.dilate(edges, kernel, iterations=1)
    edges_clean = cv2.erode(edges_clean, kernel, iterations=1)

    contours, _ = cv2.findContours(
        edges_clean,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    detected = False
    center = None
    best_contour = None

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < MIN_AREA:
            continue

        x, y, bw, bh = cv2.boundingRect(contour)
        aspect_ratio = bw / float(bh)

        # Libro abierto visto desde arriba suele verse ancho
        if aspect_ratio < 1.2 or aspect_ratio > 4.5:
            continue

        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)

        # Permitimos más de 4 puntos porque un libro abierto no siempre es perfecto
        if len(approx) < 4:
            continue

        roi_book = gray[y:y + bh, x:x + bw]
        roi_edges = edges[y:y + bh, x:x + bw]

        # Buscar línea vertical central aproximada: lomo del libro
        lines = cv2.HoughLinesP(
            roi_edges,
            rho=1,
            theta=np.pi / 180,
            threshold=50,
            minLineLength=int(bh * 0.4),
            maxLineGap=20
        )

        has_center_line = False

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]

                dx = abs(x2 - x1)
                dy = abs(y2 - y1)

                # Línea casi vertical
                if dy > dx * 2:
                    line_x = (x1 + x2) // 2

                    # Debe estar cerca del centro del bounding box
                    if abs(line_x - bw // 2) < bw * 0.25:
                        has_center_line = True
                        cv2.line(
                            output,
                            (x + x1, y + y1),
                            (x + x2, y + y2),
                            (255, 0, 0),
                            3
                        )
                        break

        if not has_center_line:
            continue

        M = cv2.moments(contour)

        if M["m00"] == 0:
            continue

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        detected = True
        center = (cx, cy)
        best_contour = contour
        break

    if detected and best_contour is not None:
        x, y, bw, bh = cv2.boundingRect(best_contour)

        cv2.drawContours(output, [best_contour], -1, (0, 255, 0), 3)
        cv2.rectangle(output, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
        cv2.circle(output, center, 7, (0, 0, 255), -1)

        cv2.putText(
            output,
            "LIBRO ABIERTO DETECTADO",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            3
        )

        cv2.putText(
            output,
            f"Centro: {center}",
            (30, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    else:
        cv2.putText(
            output,
            "NO SE DETECTA LIBRO ABIERTO",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            3
        )

    return output, edges_clean


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print("No se pudo abrir la camara.")
        return

    import time
    time.sleep(1)

    for _ in range(10):
        cap.read()

    while True:
        ret, frame = cap.read()

        if not ret:
            print("No se pudo leer frame.")
            break

        result, debug_edges = detect_open_book(frame)

        cv2.imshow("Deteccion libro abierto", result)
        cv2.imshow("Debug bordes", debug_edges)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()