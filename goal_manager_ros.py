import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, Quaternion
import numpy as np
import random
from functools import partial

class GoalManager(Node):
    def __init__(self):
        """
        Initializes the GoalManager node.
        """
        super().__init__('goal_manager_node')

        self.goals = {}  # Store {robot_id: (x_goal, y_goal, theta)}
        self.active_robot_ids = set()  # Tracks detected robots
        self.goal_publishers = {}  # Stores publishers for each robot
        self.manual_goals = {}  # Store manually set goals
        self.robot_subscribers = {}  # Track robot pose subscribers
        self.mode = "auto"  # Default mode is "auto"

        # Thresholds for goal completion
        self.distance_threshold = 0.2
        self.theta_threshold = np.radians(10)  # 10-degree tolerance

        # Timer for goal publishing
        self.create_timer(0.5, self.publish_goals)

        self.get_logger().info("GoalManager node started. Waiting for robots...")

    def update_robot_pose(self, msg, robot_id):
        """
        Receives robot positions and registers them if they are new.
        """
        if robot_id not in self.active_robot_ids:
            self.active_robot_ids.add(robot_id)
            self.goal_publishers[robot_id] = self.create_publisher(Pose, f'/robot_{robot_id}/goal', 10)
            self.get_logger().info(f"New robot detected: {robot_id}")

            # Assign a goal only if in auto mode
            if self.mode == "auto":
                self.assign_random_goal(robot_id)

        # Extract robot position and orientation
        x, y = msg.position.x, msg.position.y
        theta = self.get_yaw_from_quaternion(msg.orientation)

        # Check if robot has a goal
        if robot_id in self.goals:
            goal_x, goal_y, goal_theta = self.goals[robot_id]
            distance = np.sqrt((goal_x - x)**2 + (goal_y - y)**2)
            theta_diff = abs((goal_theta - theta + np.pi) % (2 * np.pi) - np.pi)

            # Stop publishing if goal reached
            if distance < self.distance_threshold and theta_diff < self.theta_threshold:
                self.get_logger().info(f"Robot {robot_id} reached its goal!")
                del self.goals[robot_id]  # Stop publishing goal
                
                # In auto mode, assign a new random goal
                if self.mode == "auto":
                    self.assign_random_goal(robot_id)

    def assign_random_goal(self, robot_id):
        """
        Generates and assigns a random goal within a 9x9 grid.
        """
        x_goal = random.uniform(-6, 6)
        y_goal = random.uniform(-6, 6)
        theta_goal = random.uniform(-np.pi, np.pi)

        self.goals[robot_id] = (x_goal, y_goal, theta_goal)
        self.get_logger().info(f"New random goal for Robot {robot_id}: {self.goals[robot_id]}")

    def set_manual_goal(self, robot_id, x, y, theta):
        """
        Manually sets a goal for a robot.
        """
        self.manual_goals[robot_id] = (x, y, theta)
        self.goals[robot_id] = (x, y, theta)
        self.get_logger().info(f"Manual goal set for Robot {robot_id}: {self.goals[robot_id]}")

    def switch_mode(self, mode):
        """
        Switch between manual and auto mode.
        """
        if mode in ["auto", "manual"]:
            self.mode = mode
            self.get_logger().info(f"Switched to {mode} mode.")
        else:
            self.get_logger().error("Invalid mode. Use 'auto' or 'manual'.")

    def publish_goals(self):
        """
        Publishes goals to robots.
        """
        for robot_id, goal in self.goals.items():
            if robot_id in self.goal_publishers:
                goal_msg = Pose()
                goal_msg.position.x = float(goal[0])
                goal_msg.position.y = float(goal[1])

                # Convert heading to quaternion
                theta_goal = float(goal[2])
                q = Quaternion()
                q.w = np.cos(theta_goal / 2)
                q.z = np.sin(theta_goal / 2)
                goal_msg.orientation = q

                self.goal_publishers[robot_id].publish(goal_msg)
                self.get_logger().info(f"Published goal for Robot {robot_id}: {goal}")

    def get_yaw_from_quaternion(self, orientation):
        """
        Convert a quaternion to a yaw angle (theta in radians).
        """
        q = orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y ** 2 + q.z ** 2)
        return np.arctan2(siny_cosp, cosy_cosp)

def main():
    rclpy.init()
    node = GoalManager()

    # Subscribe dynamically to all /robot_X/pose topics
    for robot_id in range(1, 11):  # Check for up to 10 robots dynamically
        topic = f'/robot_{robot_id}/pose'
        node.robot_subscribers[robot_id] = node.create_subscription(
            Pose, topic, partial(node.update_robot_pose, robot_id=robot_id), 10
        )

    # 🔹 Manually set goals here (edit this section as needed)
    node.set_manual_goal(1, 3, 5, 0.0)  # Robot 1 → (x=3.0, y=-2.0, theta=π/2)
    node.set_manual_goal(2, 3, 0, 3.14)   # Robot 2 → (x=-4.0, y=5.0, theta=0.0)
    node.set_manual_goal(3, -3, 0, 0.0)  # Robot 3 → (x=0.0, y=0.0, theta=-π/2)
    node.set_manual_goal(4, 2, -2.0, 1.57)  # Robot 1 → (x=3.0, y=-2.0, theta=π/2)
    node.set_manual_goal(5, -4.0, 5.0, 0.0)   # Robot 2 → (x=-4.0, y=5.0, theta=0.0)
    node.set_manual_goal(6, 0.0, 0.0, -1.57)  # Robot 3 → (x=0.0, y=0.0, theta=-π/2)

    # Ensure the system is running in manual mode
    node.switch_mode("manual")

    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
