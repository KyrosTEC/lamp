from so101_controller import SO101Controller


def main():
    robot = SO101Controller()
    robot.connect()

    try:
        print("Probando zonas del SO101.")
        print("Zonas:")
        print("1 | 2 | 3")
        print("4 | 5 | 6")
        print("7 | 8 | 9")
        print()
        print("Escribe un número de zona para mover.")
        print("Escribe q para salir.")

        while True:
            value = input("Zona: ").strip().lower()

            if value == "q":
                break

            if not value.isdigit():
                print("Escribe un número válido.")
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