#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Twist, TwistStamped
import torch
import torch.nn as nn
import joblib
import numpy as np
import time

# 1. ARCHITECTURE (v2 Sequential Twin - DO NOT CHANGE)
class TractionNet(nn.Module):
    def __init__(self):
        super(TractionNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(7, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 8), nn.ReLU(),
            nn.Linear(8, 4), nn.ReLU(),
            nn.Linear(4, 1), nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)

class NeuralTractionControl(Node):
    def __init__(self):
        super().__init__('neural_traction_control')

        # --- PATHS (Matches your 0.99 R2 Model) ---
        model_path = '/home/rhutvik/portfolio_projects/ros2/learning/models/traction_model_v2_20260218_163117.pth'
        scaler_path = '/home/rhutvik/portfolio_projects/ros2/learning/scalers/scaler_v2_20260218_163117.pkl'

        # Load Brain
        self.model = TractionNet()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.scaler = joblib.load(scaler_path)

        # State & Throttling
        self.imu_vals = [0.0]*6; self.v_enc = 0.0; self.last_pred = 0.0
        self.last_inf_time = 0.0

        # --- SUBSCRIPTIONS ---
        # The error was here: ensured 'self.cmd_cb' exists below
        self.create_subscription(Twist, '/cmd_vel_raw', self.cmd_cb, 10)
        self.create_subscription(Imu, '/a200_0000/sensors/imu_0/data', self.imu_cb, 10)
        self.create_subscription(Odometry, '/a200_0000/platform/odom', self.odom_cb, 10)
        
        # Publisher
        self.safe_cmd_pub = self.create_publisher(TwistStamped, '/a200_0000/cmd_vel', 10)

        self.get_logger().info("🔥 v2 High-Precision Traction Safety Node Active.")

    def imu_cb(self, msg):
        self.imu_vals = [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z,
                         msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z]

    def odom_cb(self, msg): 
        self.v_enc = msg.twist.twist.linear.x

    def cmd_cb(self, msg):
        now = time.time()
        # 20Hz Inference for stability
        if (now - self.last_inf_time) > 0.05:
            features = np.array(self.imu_vals + [self.v_enc]).reshape(1, -1)
            features_scaled = self.scaler.transform(features)
            with torch.no_grad():
                self.last_pred = self.model(torch.FloatTensor(features_scaled)).item()
            self.last_inf_time = now

        # Scaling Logic
        scaling_factor = max(0.1, 1.0 - self.last_pred)

        safe_msg = TwistStamped()
        safe_msg.header.stamp = self.get_clock().now().to_msg()
        safe_msg.header.frame_id = 'base_link'
        
        safe_msg.twist.linear.x = msg.linear.x * scaling_factor
        safe_msg.twist.angular.z = msg.angular.z

        # Debugging prints to help you see the AI working
        if self.last_pred > 0.1:
            self.get_logger().info(f"AI Detected Slip: {self.last_pred*100:.1f}% | Throttling to: {scaling_factor*100:.1f}%")
        
        self.safe_cmd_pub.publish(safe_msg)

def main():
    rclpy.init()
    node = NeuralTractionControl()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()