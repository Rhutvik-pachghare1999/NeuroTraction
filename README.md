# 🧠 NeuroTraction: Neural Network-Based Real-Time Traction Control

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![ROS2](https://img.shields.io/badge/ROS2-Humble-22314E?logo=ros&logoColor=white)](https://docs.ros.org/en/humble/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![R2 Score](https://img.shields.io/badge/R%C2%B2%20Accuracy-99.1%25-brightgreen)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**NeuroTraction** is a real-time AI-powered traction safety system for ground robots. It uses a lightweight neural network to predict wheel slip probability from live IMU and odometry data, dynamically throttling motor commands to prevent loss of traction before it occurs.

---

## 🚨 The Problem: Wheel Slip in Autonomous Ground Robots

Ground robots like the Clearpath Husky A200 operating on uneven, wet, or loose terrain are highly susceptible to wheel slip events that:
- Cause trajectory deviation and mission failure
- Damage motors and wheels through uncontrolled spinning
- Are undetectable by standard velocity controllers (PID/PD)
- Occur faster than human operator reaction time

Conventional traction controllers rely on rigid threshold rules that fail across diverse terrain types.

## 🧠 The Solution: NeuroTraction v2

NeuroTraction solves this by embedding a **TractionNet** neural network directly inside a **ROS2 safety node** that:
- Reads **7 real-time features**: 6-axis IMU (acc + gyro) + encoder velocity
- Predicts **slip probability (0–1)** at **20 Hz** inference rate
- Dynamically **scales the velocity command** before it reaches the motors
- Guarantees a minimum 10% throttle floor to avoid full stops
- Achieved **99.1% R² accuracy** on held-out test data

---

## 🏗️ System Architecture & Data Flow (DFD)

The following flowchart shows the complete real-time pipeline from sensor input to safe motor command output:

```mermaid
graph TD
    A[🤖 Clearpath Husky A200] -->|IMU 6-axis| B[sensor_msgs/Imu]
    A -->|Encoder Velocity| C[nav_msgs/Odometry]
    A -->|Operator Input| D[geometry_msgs/Twist /cmd_vel_raw]

    subgraph "Feature Engineering"
    B --> E[acc_x, acc_y, acc_z]
    B --> F[gyro_x, gyro_y, gyro_z]
    C --> G[v_enc: linear.x]
    E & F & G --> H[7-Feature Vector]
    H --> I[StandardScaler: scaler_v2.pkl]
    end

    subgraph "TractionNet Inference @ 20Hz"
    I --> J[TractionNet MLP]
    J --> K{Slip Probability}
    K -->|Low < 0.1| L[Full Throttle 100%]
    K -->|High > 0.1| M[Scale: 1.0 - slip_prob]
    end

    subgraph "ROS2 Safety Output"
    D --> N[cmd_cb]
    L & M --> N
    N --> O[TwistStamped /a200_0000/cmd_vel]
    O --> P[✅ Safe Motor Command]
    end

    style A fill:#f96,stroke:#333,stroke-width:2px
    style J fill:#bbf,stroke:#333,stroke-width:2px
    style P fill:#dfd,stroke:#333,stroke-width:2px
```

---

## 📂 Repository Structure

```bash
NeuroTraction/
├── learning/                        # Offline Training Pipeline
│   ├── models/                      # Saved TractionNet weights (.pth)
│   ├── scalers/                     # Fitted StandardScaler objects (.pkl)
│   ├── train_traction_ai.py         # V1 Training Script
│   ├── train_traction_ai_v2.py      # V2 Training Script (99.1% R²)
│   ├── evaluate_traction_ai.py      # Model evaluation & metrics
│   └── main.py                      # Dataset collection entry point
└── scripts/                         # ROS2 Deployment
    └── traction_inference.py        # Live ROS2 Safety Node
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Collect Training Data
Run your robot on diverse terrain and log IMU + Odometry topics:
```bash
python learning/main.py
```

### 3. Train the Model
```bash
python learning/train_traction_ai_v2.py
```

### 4. Evaluate Performance
```bash
python learning/evaluate_traction_ai.py
```

### 5. Deploy on Robot (ROS2)
```bash
ros2 run neurotraction traction_inference.py
```

---

## 🧠 Model Architecture: TractionNet v2

**TractionNet** is a lightweight Sequential MLP designed for low-latency embedded inference:

```
Input (7 features: acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, v_enc)
        │
        ▼
  Linear(7 → 32) + ReLU
        │
        ▼
  Linear(32 → 16) + ReLU
        │
        ▼
  Linear(16 → 8) + ReLU
        │
        ▼
  Linear(8 → 4) + ReLU
        │
        ▼
  Linear(4 → 1) + Sigmoid
        │
        ▼
  Slip Probability [0.0 → 1.0]
```

**Throttle Scaling Logic:**
```python
scaling_factor = max(0.1, 1.0 - slip_probability)
safe_velocity  = cmd_velocity * scaling_factor
```

---

## 📊 Model Performance

| Metric | V1 | V2 (Current) |
| :--- | :--- | :--- |
| **R² Score** | ~94% | **99.1%** |
| **Inference Rate** | 10 Hz | **20 Hz** |
| **Input Features** | 6 | **7** |
| **Throttle Floor** | None | **10% min** |

---

## 🛠️ Built With

- **PyTorch** — Neural network training & inference
- **ROS2 (Humble)** — Real-time robot middleware
- **scikit-learn** — Feature scaling (StandardScaler)
- **Clearpath Husky A200** — Target hardware platform

---

## 👤 Author

**Rhutvik Pachghare**  
Master’s in Robotics & Automation | Arizona State University  
[GitHub](https://github.com/Rhutvik-pachghare1999) | [LinkedIn](https://www.linkedin.com/in/rhutvik-pachghare/)
