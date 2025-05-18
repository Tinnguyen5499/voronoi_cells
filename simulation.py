import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, Twist, Quaternion
from robot import Robot
from std_msgs.msg import String  # For receiving Voronoi diagram updates
from goal_manager import GoalManager
import time

class Simulation(Node):
    def __init__(self, dt=0.01, robot_ids=[1]):
        """
        Initialize the simulation environment for multiple robots.
        :param dt: Time step for updates
        :param robot_ids: List of robot IDs (e.g., [1, 2, 3])
        """
        super().__init__('simulation_node')
        self.dt = dt
        self.robots = {robot_id: Robot(x=0, y=0, theta=0) for robot_id in robot_ids}
        self.robot_ids = robot_ids
        self.goal_manager = GoalManager()  # ✅ Initialize GoalManager

        # ✅ ROS Communication Setup
        self.pose_publishers = {}
        self.cmd_vel_subscribers = {}

        for robot_id in self.robot_ids:
            self.pose_publishers[robot_id] = self.create_publisher(
                Pose, f'/robot_{robot_id}/pose', 10
            )

            # ✅ Subscribe to velocity commands from /robot_X/cmd_vel (NOT Voronoi)
            self.cmd_vel_subscribers[robot_id] = self.create_subscription(
                Twist, f'/robot_{robot_id}/cmd_vel', 
                lambda msg, rid=robot_id: self.cmd_vel_callback(msg, rid), 10
            )


        self.timer = self.create_timer(dt, self.publish_poses)

    def set_goal(self, robot_id, goal):
        """ Proxy method to set goal via GoalManager """
        self.goal_manager.set_goal(robot_id, goal)

    def cmd_vel_callback(self, msg, robot_id):
        """ Handle velocity commands with speed limits (from Voronoi controller). """
        v = np.clip(msg.linear.x, -self.robots[robot_id].max_linear_speed, self.robots[robot_id].max_linear_speed)
        omega = np.clip(msg.angular.z, -self.robots[robot_id].max_angular_speed, self.robots[robot_id].max_angular_speed)

        L = self.robots[robot_id].L  # Wheelbase

        v_l = v - (L / 2) * omega
        v_r = v + (L / 2) * omega

        self.robots[robot_id].set_velocity(v_l, v_r)

    def publish_poses(self):
        """ ✅ Publish each robot's state as Pose to `/robot_N/pose`. """
        for robot_id, robot in self.robots.items():
            pose_msg = Pose()
            pose_msg.position.x = robot.x
            pose_msg.position.y = robot.y
            pose_msg.position.z = 0.0

            # Convert theta to quaternion
            q = Quaternion()
            q.w = np.cos(robot.theta / 2)
            q.z = np.sin(robot.theta / 2)
            pose_msg.orientation = q

            self.pose_publishers[robot_id].publish(pose_msg)


    def run(self):
        """ Main simulation loop to update robot motion based on cmd_vel. """
        try:
            while rclpy.ok():
                for robot_id in self.robot_ids:
                    self.robots[robot_id].update(self.dt)  # ✅ Update robot position

                self.publish_poses()  # ✅ Publish updated positions
                time.sleep(self.dt)  # ✅ Maintain loop timing
        except KeyboardInterrupt:
            self.get_logger().info("Simulation stopped by user.")



