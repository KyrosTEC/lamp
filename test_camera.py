import cv2

for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        ret, frame = cap.read()
        print(f"Camara {i}: abierta, frame={ret}")
        cap.release()
    else:
        print(f"Camara {i}: no abre")