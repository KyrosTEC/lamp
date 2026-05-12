import time

from so101_controller import SO101Controller


JOINT = "shoulder_pan.pos"


def move_joint(robot, joint, delta, duration=2.0, steps=80):
    start = robot.get_pose()

    if joint not in start:
        print(f"No existe el joint: {joint}")
        print("Joints disponibles:")
        for key in start.keys():
            if ".pos" in key:
                print(key)
        return

    start_value = start[joint]
    target_value = start_value + delta

    print("===================================")
    print(f"Probando joint: {joint}")
    print(f"Inicio: {start_value}")
    print(f"Objetivo: {target_value}")
    print(f"Delta: {delta}")
    print("===================================")

    for i in range(steps + 1):
        alpha = i / steps
        action = robot.get_pose()

        action[joint] = start_value + alpha * (target_value - start_value)

        robot.robot.send_action(action)
        time.sleep(duration / steps)

    time.sleep(0.5)

    end = robot.get_pose()

    print("Resultado final:")
    print(f"{joint}: {end[joint]}")


def main():
    robot = SO101Controller()
    robot.connect()

    try:
        while True:
            print()
            print("Prueba shoulder_pan.pos")
            print("1 = mover +10 grados")
            print("2 = mover -10 grados")
            print("3 = mover +25 grados")
            print("4 = mover -25 grados")
            print("q = salir")

            option = input("Opción: ").strip().lower()

            if option == "q":
                break

            if option == "1":
                move_joint(robot, JOINT, 10)

            elif option == "2":
                move_joint(robot, JOINT, -10)

            elif option == "3":
                move_joint(robot, JOINT, 25)

            elif option == "4":
                move_joint(robot, JOINT, -25)

            else:
                print("Opción inválida.")

    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()