# Real-Time Object Detection for Autonomous Robots using YOLOv8

This project implements the vision and control logic for an autonomous robot. It uses YOLOv8 to perceive the environment through a webcam and generates control commands (Forward, Stop, Turn) in real-time.

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Project

### Option 1: Autonomous Robot Mode (Full Project)
This runs the full system with robot decision logic (as per your poster).
```bash
python3 main.py
```

### Option 2: Basic Real-Time Detection
If you only want to demonstrate the object detection capability without robot commands:
```bash
python3 basic_detect.py
```

A window will open showing the robot's "eye view" with an overlay dashboard displaying:
- **Detected Objects** (Green boxes for target).
- **Control Commands** (MOVE FORWARD, TURN LEFT, STOP).
- **System FPS**.

## Project Components
- `main.py`: The core real-time application using live webcam feed.
- `benchmark_fps.py`: Utility to measure model performance (used for poster results).
- `robot_control_sim.py`: A simulation version using a video file (for testing without a camera).

## Results
The system typically achieves **~200 FPS** on NVIDIA RTX 3050 GPUs (using YOLOv8n), making it highly suitable for high-speed robotics.
