import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


REAL_COLUMNS = [
    "shoulder_pan_real",
    "shoulder_lift_real",
    "elbow_flex_real",
    "wrist_flex_real",
    "wrist_roll_real",
    "gripper_real",
]

CMD_COLUMNS = [
    "shoulder_pan_cmd",
    "shoulder_lift_cmd",
    "elbow_flex_cmd",
    "wrist_flex_cmd",
    "wrist_roll_cmd",
    "gripper_cmd",
]

ERROR_COLUMNS = [
    "shoulder_pan_error",
    "shoulder_lift_error",
    "elbow_flex_error",
    "wrist_flex_error",
    "wrist_roll_error",
    "gripper_error",
]


def prepare_dataframe(csv_path):
    df = pd.read_csv(csv_path)

    numeric_columns = (
        ["iteration", "max_error"]
        + REAL_COLUMNS
        + CMD_COLUMNS
        + ERROR_COLUMNS
    )

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=["iteration"])

    return df


def plot_max_error(df, output_dir):
    plt.figure()
    plt.plot(df["iteration"], df["max_error"])
    plt.xlabel("Iteration")
    plt.ylabel("Max error (deg)")
    plt.title("MPC maximum joint error over iterations")
    plt.grid(True)

    path = output_dir / "mpc_max_error.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Gráfica guardada: {path}")


def plot_real_vs_commanded_positions(df, output_dir):
    plt.figure()

    for real_col, cmd_col in zip(REAL_COLUMNS, CMD_COLUMNS):
        if real_col not in df.columns or cmd_col not in df.columns:
            continue

        joint_name = real_col.replace("_real", "")

        plt.plot(
            df["iteration"],
            df[real_col],
            label=f"{joint_name} real",
        )

        plt.plot(
            df["iteration"],
            df[cmd_col],
            linestyle="--",
            label=f"{joint_name} cmd",
        )

    plt.xlabel("Iteration")
    plt.ylabel("Joint position")
    plt.title("Real vs commanded joint positions")
    plt.grid(True)
    plt.legend(fontsize=7)

    path = output_dir / "mpc_real_vs_commanded_positions.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Gráfica guardada: {path}")


def plot_control_effort(df, output_dir):
    plt.figure()

    for cmd_col in CMD_COLUMNS:
        if cmd_col not in df.columns:
            continue

        joint_name = cmd_col.replace("_cmd", "")
        effort = df[cmd_col].diff().fillna(0)

        plt.plot(
            df["iteration"],
            effort,
            label=f"Δ{joint_name}",
        )

    plt.xlabel("Iteration")
    plt.ylabel("Control effort Δq")
    plt.title("MPC control effort per joint")
    plt.grid(True)
    plt.legend(fontsize=7)

    path = output_dir / "mpc_control_effort.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Gráfica guardada: {path}")


def generate_plots(csv_path, output_root="plots"):
    csv_path = Path(csv_path)

    if not csv_path.exists():
        print(f"ERROR: No existe el CSV: {csv_path}")
        return None

    df = prepare_dataframe(csv_path)

    if df.empty:
        print(f"WARNING: CSV vacío o inválido: {csv_path}")
        return None

    output_dir = Path(output_root) / csv_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_max_error(df, output_dir)
    plot_real_vs_commanded_positions(df, output_dir)
    plot_control_effort(df, output_dir)

    return output_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", help="Path del CSV generado por MPCLogger")
    parser.add_argument(
        "--output-dir",
        default="plots",
        help="Carpeta raíz donde se guardarán las gráficas",
    )

    args = parser.parse_args()

    generate_plots(args.csv_path, args.output_dir)


if __name__ == "__main__":
    main()