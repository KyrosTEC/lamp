# led_controller.py

import time
import serial


class LEDController:
    def __init__(self, port="/dev/ttyUSB0", baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.esp32 = None
        self.current_state = None

    def connect(self):
        try:
            self.esp32 = serial.Serial(self.port, self.baudrate, timeout=1)
            time.sleep(2)
            self.off()
            print(f"[LEDController] ESP32 conectada en {self.port}")
        except Exception as e:
            self.esp32 = None
            print(f"[LEDController] No se pudo conectar la ESP32: {e}")

    def send(self, command):
        if self.esp32 is None:
            return

        if not self.esp32.is_open:
            return

        try:
            self.esp32.write(command.encode("utf-8"))
        except Exception as e:
            print(f"[LEDController] Error enviando comando: {e}")

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