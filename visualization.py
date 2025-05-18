import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from std_msgs.msg import String  # For receiving Voronoi diagram updates
from geometry_msgs.msg import Pose  # ✅ Import Pose for goal messages
import json  # ✅ Fix: Import JSON for parsing BVC edges


class Visualization(Node):
    def __init__(self, robot_ids=[1]):
        """
        Initialize the visualization node for multiple robots.
        :param robot_ids: List of robot IDs (e.g., [1, 2, 3])
        """
        super().__init__('visualization_node')
        self.robot_ids = robot_ids
        self.robots = {robot_id: {'x': 0, 'y': 0} for robot_id in robot_ids}
        self.goals = {robot_id: None for robot_id in robot_ids}  # ✅ Store goals

        # ✅ Subscribe to robot goal updates from GoalManager
        for robot_id in self.robot_ids:
            self.create_subscription(
                Pose, f'/robot_{robot_id}/goal',
                lambda msg, rid=robot_id: self.update_goal(msg, rid), 10
            )

        # ✅ Initialize Plot
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.ax.set_xlim(-10, 10)
        self.ax.set_ylim(-10, 10)
        self.ax.set_xlabel("X Position (m)")
        self.ax.set_ylabel("Y Position (m)")
        self.ax.set_title("Multi-Robot Simulation")
        self.ax.grid()

        # ✅ Initialize Voronoi storage
        self.voronoi_edges = []  # List to store standard Voronoi edges
        self.bvc_edges = []  # List to store Buffered Voronoi edges

        # ✅ Subscribe to robot poses and Voronoi updates
        self.pose_subscribers = {}
        for robot_id in self.robot_ids:
            self.pose_subscribers[robot_id] = self.create_subscription(
                Pose, f'/robot_{robot_id}/pose',
                lambda msg, rid=robot_id: self.update_position(msg, rid), 10
            )

        # ✅ Subscribe to Voronoi and Buffered Voronoi Cells (BVCs)
        self.create_subscription(String, "/voronoi_edges", self.update_voronoi_graph, 10)
        self.create_subscription(String, "/buffered_voronoi_edges", self.update_bvc_graph, 10)  # ✅ BVC Subscription

        # ✅ Remove ROS timer to avoid Matplotlib threading issues

    def update_goal(self, msg, robot_id):
        """ ✅ Store received goal position and heading from GoalManager """
        theta = self.get_yaw_from_quaternion(msg.orientation)  # ✅ Extract `theta`
        self.goals[robot_id] = (msg.position.x, msg.position.y, theta)  # ✅ Store `theta`
        self.get_logger().info(f"Goal updated for Robot {robot_id}: {self.goals[robot_id]}")


    def get_yaw_from_quaternion(self, orientation):
        """ Convert a quaternion to a yaw angle (theta in radians) """
        q = orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y ** 2 + q.z ** 2)
        return np.arctan2(siny_cosp, cosy_cosp)


    def update_position(self, msg, robot_id):
        """ Update robot position and heading from Pose messages """
        self.robots[robot_id]['x'] = msg.position.x
        self.robots[robot_id]['y'] = msg.position.y

        # ✅ Extract heading angle (theta) from quaternion
        self.robots[robot_id]['theta'] = self.get_yaw_from_quaternion(msg.orientation)

        self.get_logger().info(f"Robot {robot_id} updated position: {self.robots[robot_id]}")


    def get_shape(self, x, y, theta):
        """ Generate a rectangular shape similar to the TurtleBot 2, correctly oriented. """
        length = 0.35  # Robot length (m)
        width = 0.30   # Robot width (m)
        front_extension = 0.05  # Small front tip for heading indication

        # Define base polygon with front indicator
        points = np.array([
            [-width / 2, -length / 2],  # Bottom-left
            [-width / 2, length / 2],   # Top-left
            [0, length / 2 + front_extension],  # Front indicator
            [width / 2, length / 2],    # Top-right
            [width / 2, -length / 2]    # Bottom-right
        ])

        # Rotate according to theta
        theta_adjusted = theta - np.pi / 2
        
        rotation_matrix = np.array([
            [np.cos(theta_adjusted), -np.sin(theta_adjusted)],
            [np.sin(theta_adjusted), np.cos(theta_adjusted)]
        ])
        rotated_points = points @ rotation_matrix.T

        # Translate to robot position
        rotated_points[:, 0] += x
        rotated_points[:, 1] += y

        return rotated_points


    def update_voronoi_graph(self, msg):
        """ Receive Voronoi graph edges from Voronoi Controller """
        try:
            # Use json.loads instead of eval for safety
            import json
            received_edges = json.loads(msg.data)
            
            # Validate the data format
            if isinstance(received_edges, list):
                self.voronoi_edges = received_edges
                self.get_logger().info(f"Updated Voronoi with {len(self.voronoi_edges)} edges")
            else:
                self.get_logger().error("Received invalid Voronoi data format")
        except Exception as e:
            self.get_logger().error(f"Failed to parse Voronoi edges: {e}")


    def update_bvc_graph(self, msg):
        """ Receive Buffered Voronoi Cell (BVC) edges """
        try:
            received_edges = json.loads(msg.data)
            if isinstance(received_edges, list):
                self.bvc_edges = received_edges
                self.get_logger().info(f"Updated BVC with {len(self.bvc_edges)} edges")
            else:
                self.get_logger().error("Received invalid BVC data format")
        except Exception as e:
            self.get_logger().error(f"Failed to parse BVC edges: {e}")

    def update_plot(self):
        """ Update the visualization with robot positions and Voronoi edges """
        self.ax.clear()
        self.ax.set_xlim(-10, 10)
        self.ax.set_ylim(-10, 10)
        self.ax.set_xlabel("X Position (m)")
        self.ax.set_ylabel("Y Position (m)")
        self.ax.set_title("Multi-Robot Simulation")
        self.ax.grid()

        # ✅ Fetch and draw Voronoi edges
        try:
            for edge in self.voronoi_edges:
                self.ax.plot([edge[0], edge[2]], [edge[1], edge[3]], 'k--', alpha=0.6)
        except Exception as e:
            self.get_logger().error(f"Error drawing Voronoi edges: {e}")

        # ✅ Draw Buffered Voronoi Cells (Green Solid Lines)
        try:
            for edge in self.bvc_edges:
                self.ax.plot([edge[0], edge[2]], [edge[1], edge[3]], 'g-', alpha=0.8, linewidth=2)
        except Exception as e:
            self.get_logger().error(f"Error drawing BVC edges: {e}")

        # ✅ Draw Goals using the same shape as the robot but in green
        for robot_id, goal in self.goals.items():
            if goal is not None:
                x, y, theta = goal  # ✅ Extract `theta`
                shape = self.get_shape(x, y, theta)  # ✅ Use the same function as the robot
                goal_patch = patches.Polygon(shape, closed=True, color='g', alpha=0.6)  # ✅ Green color
                self.ax.add_patch(goal_patch)

        # ✅ Draw Robots using `get_shape()`
        for robot_id, pos in self.robots.items():
            if pos['x'] is not None and pos['y'] is not None:
                theta = pos.get('theta', 0)  # Default to 0 if not available
                x, y = pos['x'], pos['y']

                # ✅ Get pentagon shape
                shape = self.get_shape(x, y, theta)

                # ✅ Draw the pentagon
                robot_patch = patches.Polygon(shape, closed=True, color='r', alpha=0.6)
                self.ax.add_patch(robot_patch)


        self.fig.canvas.draw()  # ✅ Replace plt.pause() with draw()


