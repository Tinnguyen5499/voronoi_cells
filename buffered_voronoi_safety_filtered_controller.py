import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, Twist
from std_msgs.msg import String
import numpy as np
import json
import shapely.geometry as geom
import shapely.ops
import scipy.io
from scipy.interpolate import RegularGridInterpolator


def angle_wrap(angle):
    return (angle + np.pi) % (2*np.pi) - np.pi

class BVCController(Node):
    def __init__(self, robot_id, excluded_robot_id=None):
        # Give a unique name per robot
        super().__init__(f'bvc_controller_node_{robot_id}')
        self.robot_id = robot_id

        self.excluded_robot_id = excluded_robot_id  # ⬅️ Store excluded robot ID

        if self.robot_id == self.excluded_robot_id:
            self.get_logger().info(f"🚫 Robot {self.robot_id} is excluded from control.")
            return  # ⬅️ Exit early, no control logic for this robot


        self.safety_params = self.load_safety_data('BRT_data_big.mat')


        # ✅ Initialize special robot tracking
        self.special_robot_pose = None
        self.special_robot_velocity = None

        if self.excluded_robot_id is not None:
            self.special_robot_sub = self.create_subscription(
                Pose,
                f'/robot_{self.excluded_robot_id}/pose',
                self.special_robot_callback,
                10
            )

        self.current_pose = None
        self.goal_pose = None
        self.bvc_polygon = None
        self.max_linear_speed = 0.26
        self.max_angular_speed = 1.82
        self.arrival_tolerance = 0.1
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
        self.create_timer(0.01, self.control_loop)
        self.get_logger().info(f"[Robot {self.robot_id}] BVCController initialized.")

    def load_safety_data(self, file_path):
        """ Load safety parameters from a MATLAB .mat file with logs. """
        self.get_logger().info(f"📦 Loading safety data from {file_path} ...")
        
        mat_data = scipy.io.loadmat(file_path)
        
        g = mat_data['g']
        g_min = g['min'][0][0].flatten()
        g_max = g['max'][0][0].flatten()
        g_N = g['N'][0][0].flatten()
        g_axis = g['axis'][0][0].flatten()
        g_dx = g['dx'][0][0].flatten()

        self.get_logger().info(
            f"🧭 Grid loaded:\n"
            f"  min: {g_min}\n"
            f"  max: {g_max}\n"
            f"  N: {g_N}\n"
            f"  dx: {g_dx}"
        )

        vfunc = mat_data['vfunc']
        self.get_logger().info(
            f"📈 Value function shape: {vfunc.shape}, min: {np.min(vfunc):.2f}, max: {np.max(vfunc):.2f}"
        )

        # Extract controller struct
        safety_ctrl_struct = mat_data['safety_controller']
        if isinstance(safety_ctrl_struct, np.ndarray) and safety_ctrl_struct.shape == (1, 1):
            safety_ctrl_struct = safety_ctrl_struct[0, 0]

        v = np.array(safety_ctrl_struct['v'])
        w = np.array(safety_ctrl_struct['w'])

        self.get_logger().info(
            f"✅ Safety controller loaded: v shape {v.shape}, w shape {w.shape}"
        )

        return {
            'g': {
                'min': g_min,
                'max': g_max,
                'N': g_N,
                'axis': g_axis,
                'dx': g_dx,
            },
            'vfunc': vfunc,
            'safety_controller': {
                'v': v,
                'w': w,
            }
        }


    def global_to_rel_state(self, xA, xB):
        dx = xB[0] - xA[0]
        dy = xB[1] - xA[1]
        dtheta = angle_wrap(xB[2] - xA[2])  # ✅ Use angle_wrap directly

        rot_mat = np.array([
            [np.cos(xA[2]), np.sin(xA[2])],
            [-np.sin(xA[2]), np.cos(xA[2])]
        ])
        rel_xy = rot_mat @ np.array([dx, dy])

        return np.array([rel_xy[0], rel_xy[1], dtheta])



    def snap_to_grid(self, state):
        """ Snap state to the nearest grid point using g.axis. """
        g = self.safety_params['g']
        state_up = g['axis'][1::2]  # Upper bounds from [1x6]
        state_lo = g['axis'][0::2]  # Lower bounds from [1x6]

        return np.clip(state, state_lo, state_up)



    def get_safety_controller(self, x_curr, u_nom):
        """
        Python equivalent of MATLAB's get_safety_controller.m

        Args:
            x_curr (np.ndarray): Relative state [x, y, theta]
            u_nom (dict): Nominal control {'v': float, 'w': float}

        Returns:
            u_filtered (dict): {'v': ..., 'w': ...}
            safety_override (bool): True if override applied, False otherwise
        """
        params = self.safety_params
        epsilon = 0.0 # Safety threshold

        # Evaluate the value function at the current relative state
        V = self.eval_u(params['g'], params['vfunc'], x_curr)

        if V <= epsilon:
            # Safety override triggered, use safe controls
            u_safe = self.eval_u(params['g'], params['safety_controller']['v'], x_curr)
            w_safe = self.eval_u(params['g'], params['safety_controller']['w'], x_curr)

            u_filtered = {'v': u_safe, 'w': w_safe}
            safety_override = True

            self.get_logger().warn(
                f"🚨 [Robot {self.robot_id}] Safety override! Danger: {V:.2f}, "
                f"v: {u_safe:.2f}, w: {w_safe:.2f}"
            )
        else:
            # No override, use nominal controls
            u_filtered = {'v': u_nom['v'], 'w': u_nom['w']}
            safety_override = False

            self.get_logger().info(
                f"✅ [Robot {self.robot_id}] Safe. Danger: {V:.2f}, "
                f"v: {u_nom['v']:.2f}, w: {u_nom['w']:.2f}"
            )

        return u_filtered, safety_override


    def eval_u(self, gs, datas, xs, method='linear'):
        """
        Python version of Mo Chen's eval_u.m
        - Option 1: single grid, single value function, multiple states
        - Option 2: single grid, multiple value functions, one state
        - Option 3: multiple grids, multiple functions, multiple states
        """
        if isinstance(gs, dict) and isinstance(datas, np.ndarray) and isinstance(xs, np.ndarray):
            # Option 1: Single grid, single value function, multiple states
            return self.eval_u_single(gs, datas, xs, method)

        elif isinstance(gs, dict) and isinstance(datas, list) and isinstance(xs, (list, np.ndarray)):
            # Option 2: Single grid, multiple value functions, one state
            xs = np.array(xs)
            return [self.eval_u_single(gs, data_i, xs, method) for data_i in datas]

        elif isinstance(gs, list) and isinstance(datas, list) and isinstance(xs, list):
            # Option 3: list of grids, data, and query points
            return [self.eval_u_single(g, d, np.array(x), method) for g, d, x in zip(gs, datas, xs)]

        else:
            raise ValueError("Unrecognized input pattern to eval_u().")


    def eval_u_single(self, g, data, x, method='linear'):
        """
        Interpolates a value function at 3D point(s) x, similar to Mo Chen's eval_u_single.
        Handles:
        - grid vs mismatch (transpose)
        - periodicity in theta
        """
        from scipy.interpolate import RegularGridInterpolator

        # Make sure x is a 2D array (N, 3)
        x = np.array(x)
        if x.ndim == 1:
            x = x.reshape(1, -1)

        if x.shape[1] != 3:
            x = x.T  # Transpose if needed

        axis = g['axis']
        N = g['N']

        # Define grid vectors (equivalent to g.vs in MATLAB)
        x_vals = np.linspace(axis[0], axis[1], N[0])
        y_vals = np.linspace(axis[2], axis[3], N[1])
        theta_vals = np.linspace(axis[4], axis[5], N[2])

        # ⬇️ Handle periodicity for theta (last column of x)
        theta_min, theta_max = axis[4], axis[5]
        period = theta_max - theta_min
        x[:, 2] = ((x[:, 2] - theta_min) % period) + theta_min

        # Interpolate
        interpolator = RegularGridInterpolator(
            (x_vals, y_vals, theta_vals),
            data,
            method=method,
            bounds_error=False,
            fill_value=None
        )

        v = interpolator(x)
        return v.squeeze()



    def special_robot_callback(self, msg):
        """ Callback to store the pose of the excluded/special robot. """
        self.special_robot_pose = msg  # ✅ Store pose
        self.get_logger().debug(f"📍 Special Robot {self.excluded_robot_id} Pose Updated: "
                                f"({msg.position.x:.2f}, {msg.position.y:.2f})")


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
            return        # Current pose

        x = self.current_pose.position.x
        y = self.current_pose.position.y
        theta = self.get_yaw(self.current_pose)

        # Goal
        xg = self.goal_pose.position.x
        yg = self.goal_pose.position.y
        dx = xg - x
        dy = yg - y
        dist_goal = np.hypot(dx, dy)
        '''
        # If we're close enough, optionally align final heading
        if dist_goal < self.arrival_tolerance:
            final_heading_err = angle_wrap(self.get_yaw(self.goal_pose) - theta)
            if abs(final_heading_err) < self.heading_tolerance:
                self.publish_stop()
            else:
                self.rotate_in_place(final_heading_err)
            return
        '''
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
        '''
        # [FORWARD-ONLY FIX] => if heading is reversed >90°, rotate first
        if abs(heading_err) > np.pi/2:
            self.rotate_in_place(heading_err)
            return

        # If heading_err is large (> 90°), rotate in place to avoid going in reverse
        if abs(heading_err) > np.pi/2:
            self.rotate_in_place(heading_err)
            return
        '''
        # Otherwise, drive forward
        linear_speed = 1.0 * target_dist
        angular_speed = 1.5 * heading_err
        # Clip
        linear_speed  = np.clip(linear_speed, 0.0, self.max_linear_speed)
        angular_speed = np.clip(angular_speed, -self.max_angular_speed, self.max_angular_speed)

        # ✅ Safety Controller Check
        if self.special_robot_pose is not None:
            x_special = [
                self.special_robot_pose.position.x,
                self.special_robot_pose.position.y,
                self.get_yaw(self.special_robot_pose)
            ]
            x_self = [x, y, theta]
            # Convert to relative state
            x_rel = self.global_to_rel_state(x_self, x_special)

            # 🔍 Log both global positions (A = self, B = special)
            self.get_logger().info(
                f"[Robot {self.robot_id}] Global A (self):    x = {x_self[0]:.2f}, y = {x_self[1]:.2f}, θ = {x_self[2]:.2f}"
            )
            self.get_logger().info(
                f"[Robot {self.robot_id}] Global B (special): x = {x_special[0]:.2f}, y = {x_special[1]:.2f}, θ = {x_special[2]:.2f}"
            )
            self.get_logger().info(
                f"[Robot {self.robot_id}] Relative State (B w.r.t A): dx = {x_rel[0]:.2f}, dy = {x_rel[1]:.2f}, dθ = {x_rel[2]:.2f}"
            )


            # Define default motion before safety override
            u_nom = {'v': linear_speed, 'w': angular_speed}

            # Check if safety override should be applied
            x_rel_snapped = self.snap_to_grid(x_rel)
            u_filtered, safety_override = self.get_safety_controller(x_rel_snapped, u_nom)


            if safety_override:
                self.get_logger().warn(
                    f"🚨 [Robot {self.robot_id}] Safety Override ACTIVATED! "
                    f"New Velocities -> v: {u_filtered['v']:.2f}, w: {u_filtered['w']:.2f}"
                )

            linear_speed  = u_filtered['v']
            angular_speed = u_filtered['w']
            cmd = Twist()
            cmd.linear.x = np.clip(linear_speed, 0.0, self.max_linear_speed)
            cmd.angular.z =np.clip(angular_speed, -self.max_angular_speed, self.max_angular_speed)
            self.cmd_pub.publish(cmd)

        return


    def rotate_in_place(self, dtheta):
        cmd = Twist()
        turn_rate = 0.5
        cmd.angular.z = turn_rate if dtheta > 0 else -turn_rate
        self.cmd_pub.publish(cmd)

    def publish_stop(self):
        self.cmd_pub.publish(Twist())

    def get_yaw(self, pose_msg):
        q = pose_msg.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return angle_wrap(np.arctan2(siny_cosp, cosy_cosp))


def main(args=None):
    rclpy.init(args=args)
    from rclpy.executors import MultiThreadedExecutor

    robot_ids = [1,2,3,4,5,6]

    excluded_robot_id = 3  # ⬅️ Set the special robot for LQR (Change as needed)

    exec_ = MultiThreadedExecutor()
    controllers = []
    for rid in robot_ids:
        c = BVCController(rid, excluded_robot_id=excluded_robot_id)  # ⬅️ Pass exclusion
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
