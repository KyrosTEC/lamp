"""
VisionThread — hilo dedicado de captura y detección.

Ejecuta la captura de la cámara y la detección HSV en paralelo al hilo
principal de OpenCV/control. El hilo principal solo lee el resultado más
reciente usando un modelo productor/consumidor sin bloqueo.

Arquitectura:
    VisionThread (productor)  →  resultado atómico  →  main loop (consumidor)

Uso:
    vt = VisionThread(camera_index=0)
    vt.start()

    while True:
        snapshot = vt.get_latest()   # no bloquea nunca
        if snapshot is not None:
            frame, debug_mask, robot_target = snapshot
        ...

    vt.stop()
"""

import threading
import time
from collections import deque

import cv2

from vision_neon_green import detect_neon_green_paper


class VisionThread:
    """
    Hilo productor que captura y procesa frames de la cámara continuamente.

    Publica el último resultado disponible de forma thread-safe.
    El consumidor (main loop) nunca se bloquea esperando un frame.
    """

    def __init__(self, camera_index: int = 0, target_fps: float = 60.0):
        """
        Args:
            camera_index: Índice de la cámara (mismo que CAMERA_INDEX).
            target_fps:   FPS objetivo para el hilo de captura.
                          60 Hz garantiza latencia < 17 ms por frame.
        """
        self.camera_index = camera_index
        self._frame_interval = 1.0 / target_fps

        # Resultado más reciente: (result_frame, debug_mask, robot_target)
        # deque(maxlen=1) actúa como buffer atómico de un solo slot.
        self._result_buffer: deque = deque(maxlen=1)

        # Evento para señalar al hilo que debe detenerse.
        self._stop_event = threading.Event()

        # Referencia al hilo interno.
        self._thread: threading.Thread | None = None

        # Estadísticas de diagnóstico.
        self._frames_captured = 0
        self._detections = 0
        self._last_fps_time = time.monotonic()
        self._measured_fps = 0.0

        # Último frame crudo (para mostrar en UI aunque no haya detección).
        self._last_raw_frame = None
        self._frame_lock = threading.Lock()

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def start(self):
        """Inicia el hilo de captura en segundo plano."""
        if self._thread is not None and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="VisionThread",
            daemon=True,
        )
        self._thread.start()
        print("[VisionThread] Hilo iniciado.")

    def stop(self):
        """Señala al hilo que debe detenerse y espera a que termine."""
        self._stop_event.set()

        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

        print("[VisionThread] Hilo detenido.")

    def get_latest(self):
        """
        Retorna el snapshot más reciente sin bloquear.

        Returns:
            Tuple (result_frame, debug_mask, robot_target) o None si aún
            no hay ningún frame disponible.
        """
        try:
            return self._result_buffer[-1]
        except IndexError:
            return None

    def get_last_raw_frame(self):
        """Retorna el último frame BGR crudo (sin anotaciones)."""
        with self._frame_lock:
            return self._last_raw_frame

    @property
    def fps(self) -> float:
        """FPS medido del hilo de captura."""
        return self._measured_fps

    @property
    def frames_captured(self) -> int:
        return self._frames_captured

    @property
    def detections(self) -> int:
        return self._detections

    # ------------------------------------------------------------------
    # Hilo interno
    # ------------------------------------------------------------------

    def _run(self):
        cap = cv2.VideoCapture(self.camera_index)

        if not cap.isOpened():
            print("[VisionThread] ERROR: No se pudo abrir la cámara.")
            return

        # Preferir buffer pequeño para minimizar latencia
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Descartar frames iniciales de calentamiento
        for _ in range(5):
            cap.read()

        print(f"[VisionThread] Cámara abierta (index={self.camera_index}).")

        fps_frame_count = 0
        fps_window_start = time.monotonic()

        try:
            while not self._stop_event.is_set():
                loop_start = time.monotonic()

                ret, frame = cap.read()

                if not ret:
                    print("[VisionThread] WARNING: No se pudo leer frame.")
                    time.sleep(0.01)
                    continue

                self._frames_captured += 1
                fps_frame_count += 1

                # Actualizar frame crudo
                with self._frame_lock:
                    self._last_raw_frame = frame

                # Detección (HSV + contornos)
                result_frame, debug_mask, robot_target = detect_neon_green_paper(frame)

                if robot_target is not None:
                    self._detections += 1

                # Publicar resultado — el consumidor lo leerá sin bloqueo
                self._result_buffer.append((result_frame, debug_mask, robot_target))

                # Medir FPS del hilo
                now = time.monotonic()
                elapsed_fps = now - fps_window_start

                if elapsed_fps >= 2.0:
                    self._measured_fps = fps_frame_count / elapsed_fps
                    fps_frame_count = 0
                    fps_window_start = now

                # Dormir el tiempo restante del intervalo objetivo
                elapsed = time.monotonic() - loop_start
                sleep_time = self._frame_interval - elapsed

                if sleep_time > 0:
                    time.sleep(sleep_time)

        finally:
            cap.release()
            print("[VisionThread] Cámara liberada.")
