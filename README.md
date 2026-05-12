# LAMP — Optimal Control Visual Servoing with SO-101 Robot Arm

## Overview

**LAMP** is a real-time visual servoing project that integrates classical computer vision, task-space representation, Model Predictive Control (MPC), robot actuation, data logging, and automatic result visualization using an SO-101 robotic arm.

The system uses a camera to detect a fluorescent green visual target, such as a post-it or square marker. The detected image-space position is converted into a structured 3x3 visual workspace representation. Each zone is associated with a calibrated robot pose, and the SO-101 robot moves toward the selected target configuration using an MPC-based joint-space controller.

The system is designed as an end-to-end visual servoing pipeline:

```txt
Camera input
    ↓
Classical computer vision
    ↓
Visual feature extraction
    ↓
Task-space representation
    ↓
MPC controller
    ↓
Robot actuation
    ↓
Visual feedback
    ↓
CSV logging and automatic graph generation
```

---

## Project Objective

The objective of this project is to demonstrate a complete visual servoing system where a physical robot uses real-time camera feedback to regulate its motion toward a visual target.

The robot objective is to point or move toward the detected visual region in the workspace. The system converts camera measurements into control-relevant variables and uses an MPC-based control strategy to generate safe and smooth robot motion.

---

## System Description

The project detects a fluorescent green target placed on a table. The camera identifies the target, calculates its center point, and determines which region of the visual workspace contains the target.

The image is divided into a 3x3 grid:

```txt
+----------------+----------------+----------------+
|       1        |       2        |       3        |
|  top-left      |  top-center    |  top-right     |
+----------------+----------------+----------------+
|       4        |       5        |       6        |
|  middle-left   |  center        |  middle-right  |
+----------------+----------------+----------------+
|       7        |       8        |       9        |
|  bottom-left   |  bottom-center |  bottom-right  |
+----------------+----------------+----------------+
```

Each visual zone is associated with a calibrated SO-101 pose. The robot uses visual feedback to move toward the corresponding target zone while maintaining safe behavior when the target disappears or when the program exits.

---

## Main Features

- Real-time camera capture using OpenCV.
- Classical computer vision target detection.
- HSV color segmentation for fluorescent green target detection.
- Morphological filtering to reduce noise.
- Contour detection and target center extraction.
- 3x3 visual workspace grid.
- Zone-based task-space representation.
- Calibrated SO-101 poses for each visual zone.
- MPC-based joint-space control.
- Command-state MPC strategy for smoother and more reliable servo movement.
- Asynchronous robot motion using background threads.
- Safe home behavior when the target is lost or the program exits.
- Manual and automatic operation modes.
- CSV logging of MPC motion data.
- Automatic graph generation at program exit.
- Real robot actuation using LeRobot and the SO-101 follower configuration.

---

## Hardware

The physical setup includes:

- SO-101 robotic arm.
- USB camera or webcam.
- Computer running Python.
- USB serial connection to the SO-101 robot.
- Fluorescent green visual marker.
- Table workspace for visual tracking.

---

## Software Stack

The project uses:

- Python
- OpenCV
- NumPy
- Pandas
- Matplotlib
- LeRobot
- SO-101 follower robot configuration

---

## Repository Structure

Recommended project organization:

```txt
lamp/
├── main.py
├── mpc_controller.py
├── mpc_logger.py
├── so101_controller.py
├── vision_neon_green.py
├── zone_poses.py
├── README.md
├── LICENSE
│
├── logs/
│   └── *.csv
│
├── plots/
│   └── <csv_name>/
│       ├── mpc_max_error.png
│       ├── mpc_real_vs_commanded_positions.png
│       └── mpc_control_effort.png
│
├── tools/
│   ├── calibrate_zone_poses.py
│   ├── inspect_so101.py
│   ├── plot_mpc_log.py
│   ├── print_calibration_path.py
│   └── recalibrate_so101.py
│
├── tests/
│   ├── test_base_servos.py
│   ├── test_camera.py
│   ├── test_robot_move.py
│   ├── test_robot_read.py
│   ├── test_safe_home.py
│   ├── test_shoulder_pan.py
│   └── test_zones.py
│
└── legacy/
    ├── detect_open_book.py
    ├── vision_book.py
    └── tracking_zones.py
```

---

## Main Files

### `main.py`

Main execution script.

Responsibilities:

- Opens the camera.
- Reads frames in real time.
- Runs target detection.
- Draws the 3x3 tracking grid.
- Computes target center coordinates.
- Maps the target to a visual zone.
- Confirms detections across multiple frames.
- Sends zone commands to the SO-101 controller.
- Runs robot movement asynchronously so the camera does not freeze.
- Returns the robot to safe home when the target is lost.
- Generates MPC plots automatically when the program exits.
- Handles manual keyboard controls.

Keyboard controls:

```txt
r = connect robot
h = move robot to SAFE HOME
g = move robot to READY
a = toggle automatic mode
q = quit
```

---

### `vision_neon_green.py`

Computer vision module.

Responsibilities:

- Convert camera frames from BGR to HSV.
- Apply HSV thresholding for fluorescent green color detection.
- Clean the mask using morphological operations.
- Detect contours.
- Select the largest valid target contour.
- Compute bounding box.
- Compute target center.
- Return detection information and debug mask.

This module uses classical computer vision only. It does not use deep learning or neural networks.

---

### `so101_controller.py`

Robot control module.

Responsibilities:

- Connect and disconnect the SO-101 robot.
- Read current robot joint observations.
- Validate target poses.
- Execute MPC-based movement toward target poses.
- Execute ready pose.
- Execute safe home pose.
- Execute calibrated zone poses.
- Log MPC data during each motion.

Important methods:

```python
connect()
disconnect()
get_pose()
smooth_move_to_pose()
mpc_move_to_pose()
go_home()
go_ready()
go_safe_home()
go_to_zone(zone)
```

---

### `mpc_controller.py`

MPC controller module.

Responsibilities:

- Compute joint-space control errors.
- Calculate incremental joint commands.
- Apply per-joint control weights.
- Apply per-joint effort penalties.
- Apply maximum step constraints.
- Return the next command action for the robot.

The implemented model is:

```txt
q(k+1) = q(k) + u(k)
```

Where:

- `q(k)` is the current or commanded joint state.
- `u(k)` is the incremental joint command.
- `q(k+1)` is the next commanded joint state.

---

### `mpc_logger.py`

MPC logging module.

Responsibilities:

- Create one CSV file per MPC motion.
- Store real joint positions.
- Store commanded joint positions.
- Store joint errors.
- Store maximum error per iteration.
- Store iteration index and motion label.

The CSV columns are:

```txt
timestamp
iteration
zone
shoulder_pan_real
shoulder_lift_real
elbow_flex_real
wrist_flex_real
wrist_roll_real
gripper_real
shoulder_pan_cmd
shoulder_lift_cmd
elbow_flex_cmd
wrist_flex_cmd
wrist_roll_cmd
gripper_cmd
shoulder_pan_error
shoulder_lift_error
elbow_flex_error
wrist_flex_error
wrist_roll_error
gripper_error
max_error
```

---

### `tools/plot_mpc_log.py`

Plot generation script.

Responsibilities:

- Read MPC CSV logs.
- Generate error convergence plots.
- Generate real vs commanded joint position plots.
- Generate control effort plots.

Generated plots:

```txt
mpc_max_error.png
mpc_real_vs_commanded_positions.png
mpc_control_effort.png
```

When `main.py` exits, it automatically runs this script for all CSV logs generated during the current session.

---

### `zone_poses.py`

Calibration data for the 9 visual zones.

This file stores the joint-space pose associated with each target zone:

```python
ZONE_POSES = {
    1: {
        "shoulder_pan.pos": ...,
        "shoulder_lift.pos": ...,
        "elbow_flex.pos": ...,
        "wrist_flex.pos": ...,
        "wrist_roll.pos": ...,
        "gripper.pos": ...,
    },
    ...
}
```

The robot uses these calibrated poses as MPC target configurations.

---

## Classical Computer Vision Pipeline

The perception stage is based on classical computer vision.

The target is a fluorescent green object. The detection pipeline is:

```txt
Input frame
    ↓
BGR to HSV conversion
    ↓
HSV color thresholding
    ↓
Morphological filtering
    ↓
Contour extraction
    ↓
Largest contour selection
    ↓
Bounding box calculation
    ↓
Target center extraction
```

The visual feature extracted from the image is:

```txt
p = [x, y]
```

Where:

- `x` is the horizontal coordinate of the target center.
- `y` is the vertical coordinate of the target center.

---

## Task-Space Representation

The camera image is divided into 9 regions.

For an image with width `W` and height `H`:

```txt
zone_width  = W / 3
zone_height = H / 3
```

The detected target center `(x, y)` is mapped into a grid position:

```python
col = int(x // zone_width)
row = int(y // zone_height)
zone = row * 3 + col + 1
```

The resulting task variable is:

```txt
s = zone
```

Where:

```txt
s ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9}
```

This representation connects the visual feature extracted from the camera to the robot actuation layer.

---

## MPC-Based Control Formulation

The project uses Model Predictive Control as the optimal control strategy for the visual servoing task.

### State Representation

The controller uses the SO-101 joint configuration as the control state:

```txt
x_k = [
    q_1,
    q_2,
    q_3,
    q_4,
    q_5,
    q_6
]
```

Where:

```txt
q_1 = shoulder_pan.pos
q_2 = shoulder_lift.pos
q_3 = elbow_flex.pos
q_4 = wrist_flex.pos
q_5 = wrist_roll.pos
q_6 = gripper.pos
```

The vision system provides the visual target state:

```txt
p = [x, y]
```

and the task-space zone:

```txt
s ∈ {1, ..., 9}
```

The zone determines the target robot pose:

```txt
q_target = ZONE_POSES[s]
```

---

### Control Input

The control input is represented as incremental joint commands:

```txt
u_k = [
    Δq_1,
    Δq_2,
    Δq_3,
    Δq_4,
    Δq_5,
    Δq_6
]
```

The commanded state evolves as:

```txt
q_cmd(k+1) = q_cmd(k) + u_k
```

The robot receives `q_cmd(k+1)` as the next joint-space command.

---

### System Model

The controller uses a discrete-time joint-space model:

```txt
q(k+1) = q(k) + u(k)
```

This model is appropriate for the current implementation because the SO-101 is commanded through joint-position targets.

---

### Cost Function

The MPC objective minimizes joint error and control effort:

```txt
J = Σ ( ||q_k - q_target||²_Q + ||u_k||²_R )
```

Where:

- `q_k` is the current or commanded joint configuration.
- `q_target` is the calibrated target pose associated with the visual zone.
- `u_k` is the incremental joint command.
- `Q` penalizes position error.
- `R` penalizes excessive movement.

---

### Constraints

The controller considers:

- Maximum step per joint.
- Maximum relative movement target.
- Joint-space safety limits.
- Error tolerance before considering the target reached.
- Safe home behavior.
- Valid joint names and pose validation.
- Smooth trajectory requirements.

---

### Command-State MPC Strategy

The controller uses a command-state MPC strategy. Instead of always calculating the next command directly from the measured joint position, it maintains an internal commanded pose:

```txt
q_cmd
```

At every MPC iteration:

```txt
1. Read real robot pose q_real.
2. Compute real error q_target - q_real.
3. Update internal command state q_cmd.
4. Send q_cmd to the robot.
5. Log q_real, q_cmd, error, and max error.
```

This approach improves reliability for joints that move slowly under load, such as shoulder and elbow joints, because the commanded target continues progressing even if the physical joint lags behind.

---

## Closed-Loop Behavior

The system operates in closed loop using camera feedback.

The closed-loop behavior is:

1. The camera captures the current frame.
2. The vision module detects the fluorescent green target.
3. The system extracts the target center.
4. The target center is mapped into a 3x3 task-space zone.
5. The zone selects a target robot pose.
6. The MPC controller generates incremental joint commands.
7. The SO-101 moves toward the target configuration.
8. The camera continues updating the visual state.
9. If the target disappears, the robot returns to safe home.

---

## Data Logging and Automatic Graph Generation

Every MPC movement creates a CSV file in:

```txt
logs/
```

Example:

```txt
logs/20260512_000601_zone_2_ARRIBA_CENTRO.csv
```

When the program exits, `main.py` automatically generates plots for all logs created during the current execution.

Generated plots are stored in:

```txt
plots/
```

Each CSV receives its own plot folder:

```txt
plots/
└── 20260512_000601_zone_2_ARRIBA_CENTRO/
    ├── mpc_max_error.png
    ├── mpc_real_vs_commanded_positions.png
    └── mpc_control_effort.png
```

The generated graphs are:

1. **Maximum error vs iteration**

   Shows MPC convergence behavior.

2. **Real vs commanded joint positions**

   Shows how the robot joints follow the commanded trajectory.

3. **Control effort per joint**

   Shows the incremental control action `Δq`.

---

## Safe Home Behavior

A safety pose is defined to return the robot to a stable resting configuration.

The safe home pose is used when:

- The target is no longer detected.
- The user presses `h`.
- The user exits the program with `q`.
- The program is closing and the robot is still connected.

Example safe home pose:

```python
SAFE_HOME_POSE = {
    "shoulder_pan.pos": -0.8351648351648352,
    "shoulder_lift.pos": -89.0989010989011,
    "elbow_flex.pos": -0.04395604395604396,
    "wrist_flex.pos": 65.67032967032966,
    "wrist_roll.pos": -10.68131868131868,
    "gripper.pos": 85.71428571428571,
}
```

---

## How to Run

### 1. Activate the environment

```bash
conda activate lerobot
```

### 2. Connect the SO-101 robot

The default port is:

```txt
/dev/ttyACM0
```

If your robot uses a different port, update this value in `so101_controller.py`:

```python
PORT = "/dev/ttyACM0"
```

### 3. Run the project

```bash
python main.py
```

### 4. Use the controls

Inside the OpenCV window:

```txt
r = connect robot
a = enable or disable automatic mode
h = move to safe home
g = move to ready
q = quit
```

When exiting with `q`, the system automatically generates MPC graphs from the CSV logs of that session.

---

## Generate Graphs Manually

If needed, graphs can also be generated manually:

```bash
PYTHONPATH=. python tools/plot_mpc_log.py logs/<csv_file>.csv
```

Example:

```bash
PYTHONPATH=. python tools/plot_mpc_log.py logs/20260512_000601_zone_2_ARRIBA_CENTRO.csv
```

Generated plots are stored in:

```txt
plots/
```

---

## Calibration

The robot and visual zones must be calibrated before running a reliable demo.

### Calibrate zone poses

```bash
PYTHONPATH=. python tools/calibrate_zone_poses.py
```

This script records the SO-101 pose associated with each of the 9 visual zones.

### Inspect SO-101 calibration

```bash
PYTHONPATH=. python tools/inspect_so101.py
```

This script shows the robot calibration values and helps verify motor ranges.

### Recalibrate SO-101

```bash
PYTHONPATH=. python tools/recalibrate_so101.py
```

This script recalibrates the SO-101 motor ranges.

### Print calibration path

```bash
PYTHONPATH=. python tools/print_calibration_path.py
```

This script prints the path where LeRobot stores the robot calibration JSON file.

---

## Testing

Useful test scripts:

```bash
PYTHONPATH=. python tests/test_camera.py
PYTHONPATH=. python tests/test_safe_home.py
PYTHONPATH=. python tests/test_shoulder_pan.py
PYTHONPATH=. python tests/test_base_servos.py
PYTHONPATH=. python tests/test_zones.py
```

These scripts validate:

- Camera availability.
- Robot connection.
- Joint movement.
- Shoulder pan movement.
- Safe home behavior.
- Zone pose behavior.
- MPC movement toward calibrated zones.

---

## Demo Procedure

Recommended live demo flow:

1. Start the program with `python main.py`.
2. Press `r` to connect the SO-101 robot.
3. Place the fluorescent green target on the table.
4. Verify that the camera detects the target.
5. Press `a` to activate automatic mode.
6. Move the target between different visual zones.
7. Observe the detected zone on screen.
8. Observe the SO-101 moving toward the corresponding zone using MPC.
9. Remove the target and verify that the robot returns to safe home.
10. Press `q` to exit safely.
11. Open the generated plots in `plots/`.

---

## Results to Document

For the final report or presentation, document:

- Target detection success under different lighting conditions.
- Detected center coordinates.
- Zone classification accuracy.
- Robot response to different zones.
- Time required to move between zones.
- Safe home return behavior.
- Maximum error convergence.
- Real vs commanded joint positions.
- Control effort per joint.
- Closed-loop response consistency.

---

## Safety Notes

Before running automatic mode:

1. Make sure the robot has enough free space.
2. Keep hands away from the robot while it moves.
3. Test `SAFE_HOME_POSE` before automatic mode.
4. Use slow movements during testing.
5. Stop the program with `q` if the robot behaves unexpectedly.
6. Avoid forcing the robot joints while torque is active.
7. Check that the camera sees the workspace clearly.
8. Do not place fragile objects near the robot workspace.

---

## Limitations

- Color segmentation depends on lighting conditions.
- The fluorescent green target should contrast with the background.
- The camera should remain fixed during the demo.
- Zone calibration depends on the physical placement of the robot and camera.
- Robot accuracy depends on correct SO-101 motor calibration.
- The current visual target is color-based; other target types require adjusting the vision pipeline.
- The current MPC formulation uses calibrated zone poses as target configurations.

---

## Future Improvements

Possible improvements include:

- Add continuous image-based visual servoing using normalized image error.
- Add automatic HSV calibration.
- Add more advanced geometric target detection.
- Add trajectory smoothing between zone transitions.
- Add quantitative comparison between different MPC gains.
- Add real-time plot preview inside the interface.
- Add support for multiple target colors or markers.
- Compare zone-based visual servoing and continuous image-based MPC.

---

## Authors

- Oscar Carranza Hernández
- Rhett Nieto Ramírez
- Ricardo Gaspar Ochoa
- Valentina González Benedossi

---

## License

This project is distributed under the license included in this repository.
