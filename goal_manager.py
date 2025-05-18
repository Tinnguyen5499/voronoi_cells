import matplotlib.pyplot as plt
import rclpy
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Quaternion
from rclpy.node import Node
import numpy as np

class GoalManager(Node):
    def __init__(self, ax=None):
        """
        Initializes the GoalManager.
        :param ax: Matplotlib axes object for goal visualization.
        """
        super().__init__('goal_manager_node')

        self.goals = {}          # {robot_id: (x_goal, y_goal)}
        self.goal_markers = {}   # For visual markers (X marks)
        self.goal_arrows = {}     # ✅ For heading arrows at the goal
        self.paths = {}          # {robot_id: [path points]} for A* paths
        self.path_lines = {}     # Stores A* path plot lines
        self.ax = ax             # Can be None initially

        # ROS Publishers for each robot's goal
        self.goal_publishers = {}

        # 🔄 Timer to continuously publish goals (every 0.1s)
        self.publish_timer = self.create_timer(0.1, self.publish_goals)

    def set_plot(self, ax):
        """
        Set or update the Matplotlib axes after initialization.
        :param ax: Matplotlib axes object
        """
        self.ax = ax

    def set_goal(self, robot_id, goal):
        """
        Set a goal for a specific robot and update visualization.
        Publishes the goal to /robot_N/goal.
        :param robot_id: Integer representing robot ID
        :param goal: Tuple (x, y, theta) for goal position and heading
        """
        self.goals[robot_id] = goal

        # Create a publisher if not already created
        if robot_id not in self.goal_publishers:
            self.goal_publishers[robot_id] = self.create_publisher(
                Pose, f'/robot_{robot_id}/goal', 10
            )

        # ✅ Update visualization
        if self.ax:
            # Extract x, y from goal (ignore theta for scatter)
            x, y = goal[0], goal[1]
            
            if robot_id in self.goal_markers:
                self.goal_markers[robot_id].set_offsets([x, y])
            else:
                marker = self.ax.scatter(x, y, c='g', marker='X', s=100, label=f'Goal {robot_id}')
                self.goal_markers[robot_id] = marker
            
            # ✅ Add/Update Arrow for Heading
            self._update_heading_arrow(robot_id, goal)

            self.ax.legend()
            self.ax.figure.canvas.draw()

    def _update_heading_arrow(self, robot_id, goal):
        """
        Internal method to add or update heading arrow at the goal location.
        """
        x, y, theta = goal

        # ✅ Adjust heading by subtracting π/2 for correct arrow direction
        adjusted_theta = theta - np.pi

        # Arrow parameters
        arrow_length = 0.8  # Adjust for visibility
        dx = arrow_length * np.cos(adjusted_theta)
        dy = arrow_length * np.sin(adjusted_theta)

        # Remove existing arrow if it exists
        if robot_id in self.goal_arrows:
            self.goal_arrows[robot_id].remove()

        # Draw new arrow
        arrow = self.ax.arrow(
            x, y, dx, dy, head_width=0.3, head_length=0.3, fc='g', ec='g'
        )
        self.goal_arrows[robot_id] = arrow

    def publish_goals(self):
        """
        🔄 Continuously publishes all set goals to their respective topics.
        """
        for robot_id, goal in self.goals.items():
            if robot_id in self.goal_publishers:
                goal_msg = Pose()
                goal_msg.position.x = float(goal[0])
                goal_msg.position.y = float(goal[1])
                goal_msg.position.z = 0.0  # Assuming 2D plane

                # ✅ Convert heading angle to quaternion
                theta_goal = float(goal[2])  # Extract heading angle
                q = Quaternion()
                q.w = np.cos(theta_goal / 2)
                q.z = np.sin(theta_goal / 2)
                goal_msg.orientation = q

                self.goal_publishers[robot_id].publish(goal_msg)
                self.get_logger().debug(f"Published goal for Robot {robot_id}: {goal}")

                

    def set_path(self, robot_id, path):
        """
        Set A* path for a specific robot and visualize it.
        """
        self.paths[robot_id] = path

        if self.ax:
            path_x, path_y = zip(*path) if path else ([], [])

            if robot_id in self.path_lines:
                self.path_lines[robot_id].set_data(path_x, path_y)
            else:
                line, = self.ax.plot(path_x, path_y, 'm--', lw=1.5, label=f'Robot {robot_id} A* Path')
                self.path_lines[robot_id] = line

            self.ax.legend()
            self.ax.figure.canvas.draw()

    def remove_goal(self, robot_id):
        """ Remove goal for a specific robot and its visualization. """
        if robot_id in self.goals:
            del self.goals[robot_id]

        if robot_id in self.goal_markers:
            self.goal_markers[robot_id].remove()
            del self.goal_markers[robot_id]

        if robot_id in self.goal_arrows:
            self.goal_arrows[robot_id].remove()
            del self.goal_arrows[robot_id]

        self.ax.figure.canvas.draw()

    def remove_path(self, robot_id):
        """
        Remove A* path for a specific robot.
        """
        if robot_id in self.paths:
            del self.paths[robot_id]
        if robot_id in self.path_lines:
            self.path_lines[robot_id].remove()
            del self.path_lines[robot_id]
        self.ax.figure.canvas.draw()

    def get_goal(self, robot_id):
        """
        Retrieve the goal for a specific robot.
        """
        return self.goals.get(robot_id, None)

    def clear_all_goals(self):
        """
        Clear all goals and their visualizations.
        """
        self.goals.clear()
        for marker in self.goal_markers.values():
            marker.remove()
        self.goal_markers.clear()
        self.ax.figure.canvas.draw()

    def clear_all_paths(self):
        """
        Clear all A* paths and their visualizations.
        """
        self.paths.clear()
        for line in self.path_lines.values():
            line.remove()
        self.path_lines.clear()
        self.ax.figure.canvas.draw()

    def update_visualization(self):
        """ Update goal markers, heading arrows, and paths on the plot. """
        if self.ax:
            for robot_id, goal in self.goals.items():
                # Extract x, y, theta from the goal
                x, y, theta = goal
                
                # Update marker position
                if robot_id in self.goal_markers:
                    self.goal_markers[robot_id].set_offsets([x, y])
                else:
                    marker = self.ax.scatter(x, y, c='g', marker='X', s=100, 
                                        label=f'Goal {robot_id}')
                    self.goal_markers[robot_id] = marker
                
                # Update heading arrow
                self._update_heading_arrow(robot_id, goal)
