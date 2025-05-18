import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, Twist
import numpy as np
from scipy.linalg import solve_continuous_are
from rclpy.executors import MultiThreadedExecutor

class LQRController(Node):
    def __init__(self, robot_id):
        super().__init__('lqr_controller_node')

        self.robot_id = robot_id
        self.goal_pose = None
        self.current_pose = None
        self.mode = 'NAVIGATING'  # Modes: 'NAVIGATING', 'ALIGNING', 'IDLE'

        # Thresholds
        self.position_tolerance = 0.1  # Distance to goal
        self.heading_tolerance = 0.05  # Angle tolerance (rad)

        # âœ… Speed Limits
        self.max_linear_speed = 0.2   # Maximum forward speed (m/s)
        self.max_angular_speed = 1.0  # Maximum rotation speed (rad/s)


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

        self.get_logger().info(f"ðŸš€ LQR Controller started for Robot {self.robot_id}")

    def pose_callback(self, msg):
        """Update current robot pose."""
        self.current_pose = msg

    def goal_callback(self, msg):
        """Update goal pose and ensure movement continues."""
        self.goal_pose = msg
        self.mode = 'NAVIGATING'  # â¬…ï¸ Ensure it starts moving when a new goal arrives
        self.get_logger().info(
            f"ðŸŽ¯ New goal received for Robot {self.robot_id}: "
            f"({self.goal_pose.position.x}, {self.goal_pose.position.y}, "
            f"Heading: {self.get_yaw_from_quaternion(self.goal_pose.orientation):.2f} rad)"
        )


    def control_loop(self):
        """Main LQR control loop."""
        if self.current_pose is None or self.goal_pose is None:
            return

        # Robot's current state
        x = self.current_pose.position.x
        y = self.current_pose.position.y
        theta = self.get_yaw_from_quaternion(self.current_pose.orientation)

        # Goal state
        x_goal = self.goal_pose.position.x
        y_goal = self.goal_pose.position.y
        theta_goal = self.get_yaw_from_quaternion(self.goal_pose.orientation)

        # Distance to goal
        dx = x_goal - x
        dy = y_goal - y
        rho = np.hypot(dx, dy)

        # Angle difference
        dtheta = self.angle_wrap(theta_goal - theta)

        # DEBUGGING: Display key info
        self.get_logger().info(
            f"[Robot {self.robot_id}] Mode: {self.mode} | Pos: ({x:.2f}, {y:.2f}) | "
            f"Goal: ({x_goal:.2f}, {y_goal:.2f}) | Ï: {rho:.2f} | Î¸: {np.degrees(theta):.2f}Â° | Î”Î¸: {np.degrees(dtheta):.2f}Â°"
        )

        # âœ… NAVIGATING Mode: Move to (x, y)
        if self.mode == 'NAVIGATING':
            if rho < self.position_tolerance:
                self.get_logger().info(f"âœ… Robot {self.robot_id} reached the goal position.")
                self.publish_stop()
                self.mode = 'ALIGNING'  # Lock into ALIGNING mode
            else:
                self.navigate_to_goal(dx, dy, theta)

        # âœ… ALIGNING Mode: Rotate in place to desired heading
        elif self.mode == 'ALIGNING':
            if abs(dtheta) < self.heading_tolerance:
                self.get_logger().info(f"ðŸ Robot {self.robot_id} aligned to heading.")
                self.publish_stop()
            else:
                self.rotate_in_place(dtheta)

    def navigate_to_goal(self, dx, dy, theta):
        """Navigate towards the goal position using LQR."""
        rho = np.hypot(dx, dy)
        theta_goal = np.arctan2(dy, dx)
        dtheta = self.angle_wrap(theta_goal - theta)

        # State vector: [rho, dtheta]
        state = np.array([[rho], [dtheta]])

        # LQR control matrices
        A = np.array([[0, 0],
                      [0, 0]])
        B = np.array([[1, 0],
                      [0, 1]])

        Q = np.diag([10, 5])  # Higher weight on position
        R = np.diag([1, 1])   # Control effort penalty

        P = solve_continuous_are(A, B, Q, R)
        K = np.linalg.inv(R) @ B.T @ P

        # Control input: u = -K * state
        u = -K @ state

        # âœ… Clip velocities to enforce speed limits
        linear_vel = np.clip(u[0, 0], -self.max_linear_speed, self.max_linear_speed)
        angular_vel = np.clip(u[1, 0], -self.max_angular_speed, self.max_angular_speed)


        # Publish command
        cmd = Twist()
        cmd.linear.x = linear_vel
        cmd.angular.z = angular_vel
        self.cmd_vel_publisher.publish(cmd)

        self.get_logger().info(
            f"[Robot {self.robot_id}] Moving -> v: {linear_vel:.2f}, Ï‰: {angular_vel:.2f}"
        )

    def rotate_in_place(self, dtheta):
        """Rotate the robot in place using a fixed omega."""
        cmd = Twist()
        turn_rate = 0.5  # Fixed turn rate

        # Apply turn rate based on direction
        cmd.angular.z = turn_rate if dtheta > 0 else -turn_rate
        cmd.linear.x = 0.0  # No forward motion during rotation

        self.cmd_vel_publisher.publish(cmd)

        self.get_logger().info(
            f"[Robot {self.robot_id}] Rotating -> Î”Î¸: {np.degrees(dtheta):.2f}Â°, Ï‰: {cmd.angular.z:.2f}"
        )

    def publish_stop(self):
        """Stop the robot."""
        stop_cmd = Twist()
        self.cmd_vel_publisher.publish(stop_cmd)

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

    excluded_robot_id = None  # â¬…ï¸ Change to an ID (e.g., 3) to make only that robot run LQR

    robot_ids = [1, 2, 3, 4, 5, 6]  # List of all robots
    controllers = []
    executor = MultiThreadedExecutor()

    if excluded_robot_id is not None:
        # â¬‡ï¸ Run LQR only for the chosen robot, ignore all others
        controllers.append(LQRController(excluded_robot_id))
        executor.add_node(controllers[-1])
    else:
        # â¬‡ï¸ Run LQR for all robots normally
        for robot_id in robot_ids:
            controllers.append(LQRController(robot_id))
            executor.add_node(controllers[-1])

    try:
        executor.spin()
    finally:
        for ctrl in controllers:
            ctrl.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()