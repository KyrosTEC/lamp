from so101_controller import SO101Controller


def main():
    robot = SO101Controller()
    robot.connect()

    try:
        input("Presiona ENTER para mover a SAFE HOME...")
        robot.go_safe_home()

    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()