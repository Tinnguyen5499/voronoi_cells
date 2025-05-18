import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
import numpy as np
import json
import os

class DistanceMonitor(Node):
    def __init__(self, robot_ids, collision_distance=0.5, log_file="min_distance_log.json"):
        super().__init__('distance_monitor')
        self.robot_ids = robot_ids
        self.collision_distance = collision_distance  # Threshold for collision alert
        self.log_file = log_file  # File to save min distances

        # Dictionary to store robot positions
        self.robot_positions = {rid: None for rid in self.robot_ids}

        # Subscribe to each robot's pose topic
        for rid in self.robot_ids:
            self.create_subscription(
                Pose, f'/robot_{rid}/pose', lambda msg, rid=rid: self.pose_callback(msg, rid), 10
            )

        # Timer to check distances periodically
        self.create_timer(0.1, self.compute_distances)

        # Ensure the log file exists
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w') as f:
                json.dump([], f)

        self.get_logger().info("🚀 Distance Monitor Node Initialized!")

    def pose_callback(self, msg, robot_id):
        """Store the latest pose of each robot."""
        self.robot_positions[robot_id] = (msg.position.x, msg.position.y)

    def compute_distances(self):
        """Compute and log the minimum distance between all robots."""
        positions = [pos for pos in self.robot_positions.values() if pos is not None]
        if len(positions) < 2:
            return  # Not enough data to compute distances

        # Compute pairwise distances
        distances = [
            np.linalg.norm(np.array(p1) - np.array(p2))
            for i, p1 in enumerate(positions) for j, p2 in enumerate(positions) if i < j
        ]

        if distances:
            min_distance = min(distances)

            # Log the minimum distance
            log_entry = {"time": self.get_clock().now().to_msg().sec, "min_distance": min_distance}
            with open(self.log_file, 'r+') as f:
                data = json.load(f)
                data.append(log_entry)
                f.seek(0)
                json.dump(data, f, indent=4)

            self.get_logger().info(f"📏 Minimum distance: {min_distance:.2f} meters")

            # 🚨 Collision warning
            if min_distance < self.collision_distance:
                self.get_logger().warn(f"⚠️ Collision Risk! Minimum distance: {min_distance:.2f}m")

def main(args=None):
    rclpy.init(args=args)
    robot_ids = [1, 2, 3, 4, 5, 6]  # Define your robot IDs
    node = DistanceMonitor(robot_ids)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
