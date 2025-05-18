import numpy as np

class Robot:
    def __init__(self, x, y, theta, L=0.23, size=0.175, braking_factor=0.01,
                 max_linear_speed=0.26, max_angular_speed=1.82):
        """
        Initialize a TurtleBot 2-like differential drive robot.
        :param x: Initial x-position
        :param y: Initial y-position
        :param theta: Initial heading angle (radians)
        :param L: Distance between wheels (TurtleBot 2 ~ 0.23m)
        :param size: Robot radius (TurtleBot 2 ~ 0.175m)
        :param braking_factor: How quickly the robot stops when no velocity is applied
        """
        self.x = x
        self.y = y
        self.theta = theta
        self.L = L
        self.size = size
        self.v_l = 0  # Left wheel velocity
        self.v_r = 0  # Right wheel velocity
        self.braking_factor = braking_factor

        # 🚀 Speed Constraints
        self.max_linear_speed = max_linear_speed
        self.max_angular_speed = max_angular_speed

    def update(self, dt):
        """
        Update the robot's position using the ICC (Instantaneous Center of Curvature) method.
        Handles pure rotation cases and applies braking when needed.
        """
        v = (self.v_r + self.v_l) / 2  # Linear velocity
        omega = (self.v_r - self.v_l) / self.L  # Angular velocity

        # ✅ Apply speed constraints
        v = np.clip(v, -self.max_linear_speed, self.max_linear_speed)
        omega = np.clip(omega, -self.max_angular_speed, self.max_angular_speed)
        '''
        # Apply braking if both wheels are near zero
        if np.isclose(self.v_l, 0, atol=1e-4) and np.isclose(self.v_r, 0, atol=1e-3):
            self.v_l = 0
            self.v_r = 0
        else:
            self.v_l *= self.braking_factor ** dt
            self.v_r *= self.braking_factor ** dt
        '''
        # ✅ Handle pure rotation (v ≈ 0 but omega ≠ 0)
        if np.isclose(v, 0, atol=1e-4) and not np.isclose(omega, 0, atol=1e-4):
            self.theta += omega * dt  # Rotate in place
            self.theta = self.angle_wrap(self.theta)  # Keep theta within [-pi, pi]

        elif not np.isclose(omega, 0):
            # Normal ICC calculation
            R = v / omega if not np.isclose(omega, 0) else 0
            ICC_x = self.x - R * np.sin(self.theta)
            ICC_y = self.y + R * np.cos(self.theta)

            d_theta = omega * dt
            cos_dt = np.cos(d_theta)
            sin_dt = np.sin(d_theta)

            # Apply ICC transformation
            dx = cos_dt * (self.x - ICC_x) - sin_dt * (self.y - ICC_y) + ICC_x - self.x
            dy = sin_dt * (self.x - ICC_x) + cos_dt * (self.y - ICC_y) + ICC_y - self.y

            self.x += dx
            self.y += dy
            self.theta += d_theta
            self.theta = self.angle_wrap(self.theta)

        else:
            # Move straight
            self.x += v * np.cos(self.theta) * dt
            self.y += v * np.sin(self.theta) * dt

    def set_velocity(self, v_l, v_r):
        """ Set wheel velocities directly. """
        self.v_l = v_l
        self.v_r = v_r

    def get_shape(self):
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
        theta_adjusted = self.theta + np.pi / 2
        rotation_matrix = np.array([
            [np.cos(theta_adjusted), -np.sin(theta_adjusted)],
            [np.sin(theta_adjusted), np.cos(theta_adjusted)]
        ])
        rotated_points = points @ rotation_matrix.T

        # Translate to robot position
        rotated_points[:, 0] += self.x
        rotated_points[:, 1] += self.y

        return rotated_points

    def set_velocity(self, v_l, v_r):
        """ Set wheel velocities directly with constraints. """
        # ✅ Clip wheel speeds based on max limits
        v_l = np.clip(v_l, -self.max_linear_speed, self.max_linear_speed)
        v_r = np.clip(v_r, -self.max_linear_speed, self.max_linear_speed)

        self.v_l = v_l
        self.v_r = v_r

    def angle_wrap(self, angle):
        """ Wrap angle to [-pi, pi]. """
        return (angle + np.pi) % (2 * np.pi) - np.pi
