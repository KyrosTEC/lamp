import cv2
import numpy as np

CAMERA_INDEX = 0

MIN_AREA = 12000
CANNY_LOW = 100
CANNY_HIGH = 250

MESA_ANCHO_CM = 40
MESA_ALTO_CM = 30


def detect_open_book(frame):
    output = frame.copy()
    h, w = frame.shape[:2]

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
    best_metrics = None
    robot_target = None

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < MIN_AREA:
            continue

        x, y, bw, bh = cv2.boundingRect(contour)

        if bh == 0:
            continue

        aspect_ratio = bw / float(bh)
        area_ratio = area / float(h * w)

        if aspect_ratio < 1.2 or aspect_ratio > 4.5:
            continue

        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)

        if len(approx) < 4:
            continue

        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0

        roi_book = gray[y:y + bh, x:x + bw]
        roi_edges = edges[y:y + bh, x:x + bw]

        symmetry_score = 999
        mid = bw // 2

        if mid > 0:
            left = roi_book[:, :mid]
            right = roi_book[:, mid:]

            if left.size > 0 and right.size > 0:
                right_flip = cv2.flip(right, 1)
                right_flip = cv2.resize(right_flip, (left.shape[1], left.shape[0]))
                diff = cv2.absdiff(left, right_flip)
                symmetry_score = np.mean(diff)

        lines = cv2.HoughLinesP(
            roi_edges,
            rho=1,
            theta=np.pi / 180,
            threshold=50,
            minLineLength=int(bh * 0.4),
            maxLineGap=20
        )

        has_center_line = False
        vertical_lines = 0

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]

                dx = abs(x2 - x1)
                dy = abs(y2 - y1)

                if dy > dx * 2:
                    vertical_lines += 1
                    line_x = (x1 + x2) // 2

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

        x_norm = cx / w
        y_norm = cy / h

        x_cm = x_norm * MESA_ANCHO_CM
        y_cm = y_norm * MESA_ALTO_CM

        detected = True
        center = (cx, cy)
        best_contour = contour

        best_metrics = {
            "area_px": area,
            "area_ratio": area_ratio,
            "aspect_ratio": aspect_ratio,
            "solidity": solidity,
            "symmetry_score": symmetry_score,
            "vertical_lines": vertical_lines,
            "bbox": (x, y, bw, bh)
        }

        robot_target = {
            "pixel_x": cx,
            "pixel_y": cy,
            "x_norm": x_norm,
            "y_norm": y_norm,
            "x_cm": x_cm,
            "y_cm": y_cm,
            "area_px": area,
            "bbox_x": x,
            "bbox_y": y,
            "bbox_w": bw,
            "bbox_h": bh
        }

        break

    if detected and best_contour is not None:
        x, y, bw, bh = best_metrics["bbox"]

        cv2.drawContours(output, [best_contour], -1, (0, 255, 0), 3)
        cv2.rectangle(output, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
        cv2.circle(output, center, 7, (0, 0, 255), -1)

        cv2.putText(output, "LIBRO ABIERTO DETECTADO", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 3)

        cv2.putText(output, f"Pixel center: {center}", (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

        cv2.putText(output, f"Norm: x={robot_target['x_norm']:.3f}, y={robot_target['y_norm']:.3f}", (30, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

    else:
        cv2.putText(output, "NO SE DETECTA LIBRO ABIERTO", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

    return output, edges_clean, robot_target