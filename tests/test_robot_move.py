import time
from lerobot.robots.utils import make_robot_from_config
from lerobot.robots.so_follower import SO101FollowerConfig


PORT = "/dev/ttyACM0"
ROBOT_ID = "follower_kyros"


def main():
    cfg = SO101FollowerConfig(
        port=PORT,
        id=ROBOT_ID,
        max_relative_target=5.0,
    )

    robot = make_robot_from_config(cfg)
    robot.connect()

    print("Conectado.")
    print("Observacion inicial:")
    obs = robot.get_observation()
    print(obs)

    print("Enviando movimiento pequeno...")
    action = obs.copy()

    # Movimiento MUY pequeño para probar.
    # Si alguna llave no existe, no pasa nada.
    for key in action:
        if key.endswith(".pos"):
            action[key] = action[key] + 2.0

    robot.send_action(action)
    time.sleep(1.0)

    print("Observacion final:")
    print(robot.get_observation())

    robot.disconnect()
    print("Desconectado.")


if __name__ == "__main__":
    main()
