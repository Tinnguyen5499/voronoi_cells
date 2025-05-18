import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from std_msgs.msg import String
import json
import time
from functools import partial

class Visualization(Node):
    def __init__(self):
        """
        Initialize the visualization node. It actively looks for robot topics.
        """
        super().__init__('visualization_node')
        self.special_robot_id = 3  # ⬅️ Change this ID to select the special robot, or None for no special robot
        self.safety_radius = 0.5  # ⬅️ Adjust the size of the safety bubble


        self.robots = {}  # Store {robot_id: {'x': x, 'y': y, 'theta': theta}}
        self.goals = {}  # Store {robot_id: (x, y, theta)}
        self.voronoi_edges = []  # Standard Voronoi edges
        self.bvc_edges = []  # Buffered Voronoi edges
        self.active_robot_ids = set()  # Track active robots
        self.robot_subscribers = {}  # Store dynamically created subscriptions

        # Initialize Matplotlib plot
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.ax.set_xlim(-10, 10)
        self.ax.set_ylim(-10, 10)
        self.ax.set_xlabel("X Position (m)")
        self.ax.set_ylabel("Y Position (m)")
        self.ax.set_title("Multi-Robot Simulation")
        self.ax.grid()

        # Subscribe to Voronoi updates
        self.create_subscription(String, "/voronoi_edges", self.update_voronoi_graph, 10)
        self.create_subscription(String, "/buffered_voronoi_edges", self.update_bvc_graph, 10)

        # Timer to check for new robot topics every 2 seconds
        self.create_timer(2.0, self.check_for_new_robots)

        # Timer to update visualization
        self.create_timer(0.5, self.update_plot)

        self.get_logger().info("Visualization node started.")

    def check_for_new_robots(self):
        """
        Actively checks for new robots publishing on `/robot_X/pose`.
        This method ensures that all robots are visualized dynamically.
        """
        for robot_id in range(1, 11):  # Assuming robots are numbered from 1 to 10
            pose_topic = f'/robot_{robot_id}/pose'
            if robot_id not in self.active_robot_ids:
                self.create_subscription(Pose, pose_topic, partial(self.update_position, robot_id=robot_id), 10)
                self.create_subscription(Pose, f'/robot_{robot_id}/goal', partial(self.update_goal, robot_id=robot_id), 10)
                self.active_robot_ids.add(robot_id)
                self.get_logger().info(f"Subscribed to new robot: {robot_id}")

    def update_position(self, msg, robot_id):
        """ Update robot position and heading from Pose messages. """
        
        # Only add the robot if it does not already exist
        if robot_id not in self.robots:
            self.robots[robot_id] = {}

        self.robots[robot_id]['x'] = msg.position.x
        self.robots[robot_id]['y'] = msg.position.y
        self.robots[robot_id]['theta'] = self.get_yaw_from_quaternion(msg.orientation)

        self.get_logger().info(f"Robot {robot_id} updated position: {self.robots[robot_id]}")

    def update_goal(self, msg, robot_id):
        """ Store received goal position and heading. """
        self.goals[robot_id] = (msg.position.x, msg.position.y, self.get_yaw_from_quaternion(msg.orientation))
        self.get_logger().info(f"Goal updated for Robot {robot_id}: {self.goals[robot_id]}")

    def update_voronoi_graph(self, msg):
        """ Receive Voronoi graph edges. """
        try:
            self.voronoi_edges = json.loads(msg.data)
        except Exception as e:
            self.get_logger().error(f"Failed to parse Voronoi edges: {e}")

    def update_bvc_graph(self, msg):
        """ Receive Buffered Voronoi Cell (BVC) edges. """
        try:
            self.bvc_edges = json.loads(msg.data)
        except Exception as e:
            self.get_logger().error(f"Failed to parse BVC edges: {e}")

    def get_yaw_from_quaternion(self, orientation):
        """ Convert a quaternion to a yaw angle (theta in radians). """
        q = orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y ** 2 + q.z ** 2)
        return np.arctan2(siny_cosp, cosy_cosp)

    def get_shape(self, x, y, theta):
        """ Generate a rectangular shape similar to the TurtleBot 2, correctly oriented. """
        length = 0.35
        width = 0.30
        front_extension = 0.05

        points = np.array([
            [-width / 2, -length / 2],
            [-width / 2, length / 2],
            [0, length / 2 + front_extension],
            [width / 2, length / 2],
            [width / 2, -length / 2]
        ])

        theta_adjusted = theta - np.pi / 2
        rotation_matrix = np.array([
            [np.cos(theta_adjusted), -np.sin(theta_adjusted)],
            [np.sin(theta_adjusted), np.cos(theta_adjusted)]
        ])
        rotated_points = points @ rotation_matrix.T
        rotated_points[:, 0] += x
        rotated_points[:, 1] += y
        return rotated_points

    def update_plot(self):
        """ Update the visualization. """
        self.ax.clear()
        self.ax.set_xlim(-10, 10)
        self.ax.set_ylim(-10, 10)
        self.ax.set_xlabel("X Position (m)")
        self.ax.set_ylabel("Y Position (m)")
        self.ax.set_title("Multi-Robot Simulation")
        self.ax.grid()

        # Draw Voronoi edges
        for edge in self.voronoi_edges:
            self.ax.plot([edge[0], edge[2]], [edge[1], edge[3]], 'k--', alpha=0.6)

        # Draw BVC edges
        for edge in self.bvc_edges:
            self.ax.plot([edge[0], edge[2]], [edge[1], edge[3]], 'g-', alpha=0.8, linewidth=2)

        # Draw goals
        for robot_id, goal in self.goals.items():
            if goal:
                x, y, theta = goal
                shape = self.get_shape(x, y, theta)
                goal_patch = patches.Polygon(shape, closed=True, color='g', alpha=0.6)
                self.ax.add_patch(goal_patch)

        # Draw Robots
        for robot_id, pos in self.robots.items():
            x, y, theta = pos['x'], pos['y'], pos['theta']
            shape = self.get_shape(x, y, theta)

            if robot_id == self.special_robot_id:
                # ✅ Special Robot: Blue color & Safety Bubble
                robot_patch = patches.Polygon(shape, closed=True, color='b', alpha=0.6)
                safety_circle = patches.Circle((x, y), self.safety_radius, color='b', alpha=0.2, fill=True)

                self.ax.add_patch(safety_circle)  # ✅ Add safety bubble

            else:
                # ✅ Normal Robots: Red color
                robot_patch = patches.Polygon(shape, closed=True, color='r', alpha=0.6)

            self.ax.add_patch(robot_patch)

        self.fig.canvas.draw()
        plt.pause(0.001)

def main():
    rclpy.init()
    node = Visualization()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
