import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, Twist
import numpy as np
from rclpy.executors import MultiThreadedExecutor

class GreedyController(Node):
    def __init__(self, robot_id):
        super().__init__(f'greedy_controller_node_{robot_id}')
        
        self.robot_id = robot_id
        self.goal_pose = None
        self.current_pose = None
        self.mode = 'NAVIGATING'  # Modes: 'NAVIGATING', 'ALIGNING', 'IDLE'

        # Speed limits
        self.max_linear_speed = 0.26   # Max forward speed
        self.max_angular_speed = 1.82  # Max turn speed

        # Thresholds
        self.position_tolerance = 0.1  # Distance tolerance to stop
        self.heading_tolerance = 0.05  # Heading tolerance (radians)

        # Subscribers
        self.pose_subscriber = self.create_subscription(
            Pose, f'/robot_{self.robot_id}/pose', self.pose_callback, 10)

        self.goal_subscriber = self.create_subscription(
            Pose, f'/robot_{self.robot_id}/goal', self.goal_callback, 10)

        # Publisher for cmd_vel
        self.cmd_vel_publisher = self.create_publisher(
            Twist, f'/robot_{self.robot_id}/cmd_vel', 10)

        # Timer for control loop
        self.timer = self.create_timer(0.1, self.control_loop)

        self.get_logger().info(f"🚀 Greedy Controller started for Robot {self.robot_id}")

    def pose_callback(self, msg):
        """Update current robot pose."""
        self.current_pose = msg

    def goal_callback(self, msg):
        """Update goal pose and reset mode."""
        self.goal_pose = msg
        self.mode = 'NAVIGATING'  # ⬅️ Always start in NAVIGATING mode
        self.get_logger().info(
            f"🎯 New goal for Robot {self.robot_id}: "
            f"({msg.position.x}, {msg.position.y})"
        )

    def control_loop(self):
        """Move greedily towards the goal and align after reaching it."""
        if self.current_pose is None or self.goal_pose is None:
            return

        # Get current position
        x = self.current_pose.position.x
        y = self.current_pose.position.y
        theta = self.get_yaw_from_quaternion(self.current_pose.orientation)

        # Get goal position
        x_goal = self.goal_pose.position.x
        y_goal = self.goal_pose.position.y
        theta_goal = self.get_yaw_from_quaternion(self.goal_pose.orientation)

        # Compute relative distance and angle
        dx = x_goal - x
        dy = y_goal - y
        rho = np.hypot(dx, dy)  # Distance to goal
        theta_goal_direction = np.arctan2(dy, dx)  # Desired heading
        dtheta = self.angle_wrap(theta_goal_direction - theta)  # Heading error

        # ✅ NAVIGATING: Move to goal
        if self.mode == 'NAVIGATING':
            if rho < self.position_tolerance:
                self.get_logger().info(f"✅ [Robot {self.robot_id}] Reached the goal position!")
                self.publish_stop()
                self.mode = 'ALIGNING'  # ⬅️ Now align to goal heading
            else:
                self.move_to_goal(rho, dtheta)

        # ✅ ALIGNING: Rotate to match goal heading
        elif self.mode == 'ALIGNING':
            final_heading_error = self.angle_wrap(theta_goal - theta)
            if abs(final_heading_error) < self.heading_tolerance:
                self.get_logger().info(f"🏁 [Robot {self.robot_id}] Aligned to goal heading!")
                self.publish_stop()
                self.mode = 'IDLE'  # ⬅️ Stop everything
            else:
                self.rotate_to_heading(final_heading_error)

        # ✅ IDLE: Stay still
        elif self.mode == 'IDLE':
            self.publish_stop()

    def move_to_goal(self, rho, dtheta):
        """Move greedily towards the goal with speed limits."""
        if abs(dtheta) > self.heading_tolerance:
            # **Turn first** if heading error is large
            linear_vel = 0.0  # No forward motion while turning
            angular_vel = np.clip(dtheta * 2.0, -self.max_angular_speed, self.max_angular_speed)
        else:
            # **Move forward** if heading is correct
            linear_vel = np.clip(rho, 0.05, self.max_linear_speed)  # Min speed to prevent stalling
            angular_vel = np.clip(dtheta * 2.0, -self.max_angular_speed, self.max_angular_speed)

        self.publish_command(linear_vel, angular_vel)

    def rotate_to_heading(self, dtheta):
        """Rotate the robot to match goal heading."""
        angular_vel = np.clip(dtheta * 2.0, -self.max_angular_speed, self.max_angular_speed)
        self.publish_command(0.0, angular_vel)  # No forward motion during rotation

    def publish_command(self, linear, angular):
        """Publish velocity command."""
        cmd = Twist()
        cmd.linear.x = linear
        cmd.angular.z = angular
        self.cmd_vel_publisher.publish(cmd)

        self.get_logger().info(
            f"[Robot {self.robot_id}] Moving -> v: {linear:.2f}, ω: {angular:.2f}"
        )

    def publish_stop(self):
        """Stop the robot."""
        self.cmd_vel_publisher.publish(Twist())

    def get_yaw_from_quaternion(self, orientation):
        """Convert quaternion to yaw (theta)."""
        q = orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y ** 2 + q.z ** 2)
        return np.arctan2(siny_cosp, cosy_cosp)

    def angle_wrap(self, angle):
        """Wrap angle to [-pi, pi]."""
        return (angle + np.pi) % (2 * np.pi) - np.pi

def main(args=None):
    rclpy.init(args=args)

    robot_ids = [3]  # Adjust as needed
    controllers = [GreedyController(robot_id) for robot_id in robot_ids]

    executor = MultiThreadedExecutor()
    for ctrl in controllers:
        executor.add_node(ctrl)

    try:
        executor.spin()
    finally:
        for ctrl in controllers:
            ctrl.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
