from so101_controller import SO101Controller


def main():
    robot = SO101Controller()
    robot.connect()

    try:
        print("Prueba de zonas.")
        print("Usa zonas donde cambian mucho los servos de abajo.")
        print("Ejemplo: 2, 8, 3, 9, 5")
        print("q = salir")

        while True:
            value = input("Zona: ").strip().lower()

            if value == "q":
                break

            if not value.isdigit():
                print("Escribe un número de zona válido.")
                continue

            zone = int(value)

            if zone < 1 or zone > 9:
                print("La zona debe estar entre 1 y 9.")
                continue

            robot.go_to_zone(zone)

    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()