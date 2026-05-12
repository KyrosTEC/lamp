# LAMP — Optimal Control Visual Servoing with SO-101 Robot Arm

## Overview

**LAMP** is a real-time visual servoing project that integrates classical computer vision, task-space representation, Model Predictive Control (MPC), and physical robot actuation using an SO-101 robotic arm.

The system uses a camera to detect a fluorescent green visual target, such as a post-it or square marker, extracts its image-space position, maps that position into a structured visual task representation, and commands the SO-101 robot arm to move toward the corresponding target region. The project is designed as an end-to-end visual servoing pipeline in which visual feedback is used to guide robot motion.

The complete pipeline is:

```txt
Camera input
    ↓
Classical computer vision
    ↓
Visual feature extraction
    ↓
Task-space representation
    ↓
MPC control formulation
    ↓
Robot actuation
    ↓
Visual feedback
```

---

## Project Objective

The objective of this project is to demonstrate a complete visual servoing system where a robot uses real-time camera feedback to regulate its motion toward a visual target.

The robot objective is to point or move toward the detected target region using image-based information. The system converts visual measurements into control-relevant variables and uses an MPC-based control strategy to determine safe and smooth robot motion.

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

Each visual zone is associated with a calibrated SO-101 pose. The robot uses the visual feedback to move toward the target zone while maintaining a safe home behavior when the target is no longer detected.

---

## Current Features

- Real-time camera capture using OpenCV.
- Classical computer vision target detection.
- HSV color segmentation for fluorescent green target detection.
- Morphological filtering to reduce noise.
- Contour detection and target center extraction.
- 3x3 visual workspace grid.
- Zone-based task-space representation.
- Calibrated SO-101 robot poses for each visual zone.
- Smooth joint-space interpolation.
- MPC-based control structure for visual servoing.
- Safe home pose behavior.
- Manual and automatic operation modes.
- Real robot actuation using LeRobot and the SO-101 follower configuration.

---

## Hardware

The physical setup includes:

- SO-101 robotic arm.
- USB camera or webcam.
- Computer running Python.
- USB serial connection to the SO-101 robot.
- Fluorescent green target marker.
- Table workspace for visual tracking.

---

## Software Stack

The project uses:

- Python
- OpenCV
- NumPy
- LeRobot
- SO-101 follower robot configuration

---

## Repository Structure

Recommended project organization:

```txt
lamp/
├── main.py
├── so101_controller.py
├── vision_neon_green.py
├── zone_poses.py
├── README.md
├── LICENSE
│
├── tools/
│   ├── calibrate_zone_poses.py
│   ├── inspect_so101.py
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
- Returns the robot to safe home when the target is lost.
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
- Move smoothly between poses.
- Execute calibrated zone poses.
- Execute ready pose.
- Execute safe home pose.
- Provide a safe motion interface for the main visual servoing loop.

Important methods:

```python
connect()
disconnect()
get_pose()
smooth_move_to_pose()
go_home()
go_ready()
go_safe_home()
go_to_zone(zone)
```

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

The robot uses these calibrated poses to point toward the detected zone.

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

The system also computes normalized visual coordinates:

```txt
x_norm = x / frame_width
y_norm = y / frame_height
```

These variables are used as task-space information for the visual servoing process.

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

The controller is formulated around the idea of reducing visual error while producing smooth and safe robot motions.

### State Representation

A representative state vector is:

```txt
x_k = [
    e_u,
    e_v,
    q_1,
    q_2,
    q_3,
    q_4,
    q_5,
    q_6
]
```

Where:

- `e_u` is the horizontal image-space error.
- `e_v` is the vertical image-space error.
- `q_i` are the SO-101 joint positions.

### Visual Error

The desired point can be defined as the center of the image or as the center of the selected visual zone.

```txt
p_desired = [u_d, v_d]
p_detected = [u, v]
```

The visual error is:

```txt
e = p_detected - p_desired
```

This error describes how far the detected target is from the desired visual configuration.

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

Where each `Δq_i` is a change in a robot joint command.

### Cost Function

The MPC objective minimizes visual error and control effort:

```txt
J = Σ (e_kᵀ Q e_k + u_kᵀ R u_k)
```

Where:

- `Q` penalizes image-space tracking error.
- `R` penalizes excessive joint movement.
- `e_k` is the visual error at time step `k`.
- `u_k` is the control input at time step `k`.

### Constraints

The controller considers constraints such as:

- Joint limits.
- Maximum relative joint movement.
- Maximum joint velocity.
- Workspace safety limits.
- Safe home behavior.
- Smooth trajectory requirements.

### Control Behavior

The controller receives the visual target state, evaluates the motion objective, and generates safe joint-space actions for the SO-101 arm.

The visual feedback loop is:

```txt
Detect target
    ↓
Compute visual error
    ↓
Solve MPC objective
    ↓
Generate robot command
    ↓
Move SO-101
    ↓
Capture next frame
```

---

## Closed-Loop Behavior

The system operates in closed loop using camera feedback.

The closed-loop behavior is:

1. The camera captures the current frame.
2. The vision module detects the target.
3. The system extracts the target center.
4. The target is mapped to the task-space representation.
5. The MPC control logic determines the motion objective.
6. The robot moves toward the corresponding target configuration.
7. The next camera frame updates the visual feedback.
8. If the target disappears, the robot returns to safe home.

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

If your robot uses a different port, update:

```python
PORT = "/dev/ttyACM0"
```

inside `so101_controller.py`.

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

---

## Calibration

The robot and the visual zones must be calibrated before running a reliable demo.

### Calibrate zone poses

```bash
PYTHONPATH=. python tools/calibrate_zone_poses.py
```

This script is used to record the SO-101 pose associated with each of the 9 visual zones.

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
8. Observe the SO-101 robot moving toward the corresponding zone.
9. Remove the target and verify that the robot returns to safe home.
10. Press `q` to exit safely.

---

## Results to Document

For the final report or presentation, the following results can be documented:

- Target detection success under different lighting conditions.
- Detected center coordinates.
- Zone classification accuracy.
- Robot response to different zones.
- Time required to move between zones.
- Safe home return behavior.
- Visual error reduction over time.
- Control effort during motion.
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

---

## Future Improvements

Possible improvements include:

- Add continuous image-based visual servoing using normalized image error.
- Log visual error, joint positions, and control commands.
- Plot convergence behavior.
- Add real-time control effort visualization.
- Improve robustness to illumination changes.
- Add automatic HSV calibration.
- Add more advanced geometric target detection.
- Add trajectory smoothing between zone transitions.
- Add quantitative evaluation scripts.
- Compare zone-based visual servoing and continuous MPC visual servoing.

---

## Authors

- Oscar Carranza Hernández
- Rhett Nieto Ramírez
- Ricardo Gaspar Ochoa
- Valentina González Benedossi

---

## License

This project is distributed under the license included in this repository.
