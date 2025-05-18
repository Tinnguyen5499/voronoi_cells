import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, Twist
from std_msgs.msg import String
import numpy as np
import json
import shapely.geometry as geom
import shapely.ops

def angle_wrap(angle):
    return (angle + np.pi) % (2*np.pi) - np.pi

class BVCController(Node):
    def __init__(self, robot_id):
        # Give a unique name per robot
        super().__init__(f'bvc_controller_node_{robot_id}')
        self.robot_id = robot_id

        self.current_pose = None
        self.goal_pose = None
        self.bvc_polygon = None
        self.max_linear_speed = 0.2
        self.max_angular_speed = np.pi
        self.arrival_tolerance = 0.2
        self.heading_tolerance = 0.1

        # Subscribe to /robot_{id}/pose
        self.pose_sub = self.create_subscription(
            Pose,
            f'/robot_{self.robot_id}/pose',
            self.pose_callback,
            10
        )

        # Subscribe to /robot_{id}/goal
        self.goal_sub = self.create_subscription(
            Pose,
            f'/robot_{self.robot_id}/goal',
            self.goal_callback,
            10
        )

        # [CHANGED HERE] Subscribe to /robot_{id}/bvc_edges instead of /buffered_voronoi_edges
        self.bvc_sub = self.create_subscription(
            String,
            f'/robot_{self.robot_id}/bvc_edges',  # The new per-robot topic
            self.bvc_callback,
            10
        )

        # Publish cmd_vel for this robot
        self.cmd_pub = self.create_publisher(
            Twist, f'/robot_{self.robot_id}/cmd_vel', 10
        )

        # Timer for the control loop
        self.create_timer(0.1, self.control_loop)
        self.get_logger().info(f"[Robot {self.robot_id}] BVCController initialized.")

    def pose_callback(self, msg):
        self.current_pose = msg

    def goal_callback(self, msg):
        self.goal_pose = msg
        self.get_logger().info(
            f"[Robot {self.robot_id}] New goal -> ({msg.position.x:.2f}, {msg.position.y:.2f})"
        )

    def bvc_callback(self, msg):
        """Receive JSON edges for *this* robot's BVC polygon."""
        try:
            edges = json.loads(msg.data)
            self.bvc_polygon = edges
            self.get_logger().debug(
                f"[Robot {self.robot_id}] Received {len(edges)} edges in BVC."
            )
        except Exception as e:
            self.get_logger().error(f"[Robot {self.robot_id}] Error parsing BVC edges: {e}")

    def control_loop(self):
        """At each iteration, move robot inside the BVC toward the goal."""
        if self.current_pose is None or self.goal_pose is None:
            return

        if not self.bvc_polygon or len(self.bvc_polygon) < 1:
            self.publish_stop()
            return

        # Current pose
        x = self.current_pose.position.x
        y = self.current_pose.position.y
        theta = self.get_yaw(self.current_pose)

        # Goal
        xg = self.goal_pose.position.x
        yg = self.goal_pose.position.y
        dx = xg - x
        dy = yg - y
        dist_goal = np.hypot(dx, dy)

        # If we're close enough, optionally align final heading
        if dist_goal < self.arrival_tolerance:
            final_heading_err = angle_wrap(self.get_yaw(self.goal_pose) - theta)
            if abs(final_heading_err) < self.heading_tolerance:
                self.publish_stop()
            else:
                self.rotate_in_place(final_heading_err)
            return

        # Convert edges -> shapely polygon
        line_elems = []
        for e in self.bvc_polygon:
            line_elems.append(geom.LineString([(e[0], e[1]), (e[2], e[3])]))
        merged = shapely.ops.linemerge(line_elems)
        polygons = list(shapely.ops.polygonize(merged))
        if len(polygons) < 1:
            self.publish_stop()
            return

        # If multiple polygons, pick one containing current (x,y), else pick first
        bvc_poly = None
        current_pt = geom.Point(x, y)
        for p in polygons:
            if p.contains(current_pt):
                bvc_poly = p
                break
        if bvc_poly is None:
            bvc_poly = polygons[0]

        # Closest point in BVC to the goal
        goal_pt = geom.Point(xg, yg)
        if bvc_poly.contains(goal_pt):
            target_x, target_y = xg, yg
        else:
            closest_pt = bvc_poly.exterior.interpolate(
                bvc_poly.exterior.project(goal_pt)
            )
            target_x, target_y = closest_pt.x, closest_pt.y

        # Decide motion
        dx_t = target_x - x
        dy_t = target_y - y
        target_dist = np.hypot(dx_t, dy_t)
        desired_heading = np.arctan2(dy_t, dx_t)
        heading_err = angle_wrap(desired_heading - theta)

        # [FORWARD-ONLY FIX] => if heading is reversed >90°, rotate first
        if abs(heading_err) > np.pi/2:
            self.rotate_in_place(heading_err)
            return

        # If heading_err is large (> 90°), rotate in place to avoid going in reverse
        if abs(heading_err) > np.pi/2:
            self.rotate_in_place(heading_err)
            return

        # Otherwise, drive forward
        linear_speed = 1.0 * target_dist
        angular_speed = 1.5 * heading_err
        # Clip
        linear_speed  = np.clip(linear_speed, 0.0, self.max_linear_speed)
        angular_speed = np.clip(angular_speed, -self.max_angular_speed, self.max_angular_speed)

        cmd = Twist()
        cmd.linear.x = linear_speed
        cmd.angular.z = angular_speed
        self.cmd_pub.publish(cmd)

    def rotate_in_place(self, dtheta):
        cmd = Twist()
        turn_rate = 0.5
        cmd.angular.z = turn_rate if dtheta > 0 else -turn_rate
        self.cmd_pub.publish(cmd)

    def publish_stop(self):
        self.cmd_pub.publish(Twist())

    def get_yaw(self, pose_msg):
        q = pose_msg.orientation
        siny_cosp = 2*(q.w*q.z + q.x*q.y)
        cosy_cosp = 1 - 2*(q.y*q.y + q.z*q.z)
        return np.arctan2(siny_cosp, cosy_cosp)

def main(args=None):
    rclpy.init(args=args)
    from rclpy.executors import MultiThreadedExecutor

    robot_ids = [1,2,4,5,6]
    exec_ = MultiThreadedExecutor()
    controllers = []
    for rid in robot_ids:
        c = BVCController(rid)
        controllers.append(c)
        exec_.add_node(c)

    try:
        exec_.spin()
    finally:
        for c in controllers:
            c.destroy_node()
        rclpy.shutdown()

if __name__=='__main__':
    main()
