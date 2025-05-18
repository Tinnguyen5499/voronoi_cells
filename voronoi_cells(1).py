import numpy as np
import rclpy
from rclpy.node import Node
from scipy.spatial import Voronoi
from geometry_msgs.msg import Pose
from std_msgs.msg import String  # ✅ Publish Voronoi edges
import json  # ✅ Convert Voronoi edges to JSON format

class VoronoiManager(Node):
    def __init__(self):
        super().__init__('voronoi_manager')

        self.arena_size = 10
        self.robot_positions = {}  # {robot_id: (x, y)}
        self.boundary_points = np.array([
            [-15, -15], [15, -15],
            [-15, 15], [15, 15]
        ])


        # ✅ Subscribe to robot positions
        for robot_id in range(1, 7):  # Assuming 6 robots
            self.create_subscription(Pose, f'/robot_{robot_id}/pose', lambda msg, rid=robot_id: self.update_position(msg, rid), 10)

        # ✅ Publish Voronoi edges as JSON
        self.voronoi_publisher = self.create_publisher(String, '/voronoi_edges', 10)

        # ✅ Publish corrected robot positions
        self.position_publishers = {robot_id: self.create_publisher(Pose, f'/robot_{robot_id}/voronoi_position', 10) for robot_id in range(1, 7)}

        # ✅ Compute and publish Voronoi every 0.5 seconds
        self.create_timer(0.5, self.compute_and_publish_voronoi)

    def update_position(self, msg, robot_id):
        """ Update robot positions from the simulation. """
        self.robot_positions[robot_id] = (msg.position.x, msg.position.y)

    def compute_voronoi(self):
        """ Compute Voronoi diagram based on robot positions. """
        if len(self.robot_positions) < 2:
            return None

        all_positions = np.vstack([list(self.robot_positions.values()), self.boundary_points])
        return Voronoi(all_positions)

    def compute_and_publish_voronoi(self):
        """ Compute Voronoi and publish edges & corrected positions. """
        if len(self.robot_positions) < 2:
            self.get_logger().info(f"Not enough robot positions for Voronoi: {len(self.robot_positions)}")
            return
        
        # Convert dictionary to list of positions
        points = list(self.robot_positions.values())
        self.get_logger().info(f"Computing Voronoi with {len(points)} robots")
        
        # Add boundary points (to ensure Voronoi does not disappear)
        boundary_pts = [
            [-12, -12], [0, -12], [12, -12],
            [-12, 0], [12, 0],
            [-12, 12], [0, 12], [12, 12]
        ]

        all_points = np.array(points + boundary_pts)

        try:
            vor = Voronoi(all_points)
            edges = []

            # ✅ Process Voronoi edges
            for ridge_points, ridge_vertices in zip(vor.ridge_points, vor.ridge_vertices):
                if -1 not in ridge_vertices:
                    # ✅ Finite edge (keep as-is)
                    vertex1 = vor.vertices[ridge_vertices[0]]
                    vertex2 = vor.vertices[ridge_vertices[1]]
                    edges.append([float(vertex1[0]), float(vertex1[1]), 
                                float(vertex2[0]), float(vertex2[1])])
                else:
                    # ✅ Infinite edge (extend to boundary)
                    finite_vertex_index = ridge_vertices[0] if ridge_vertices[1] == -1 else ridge_vertices[1]
                    finite_vertex = vor.vertices[finite_vertex_index]

                    # ✅ Compute correct perpendicular direction
                    p1, p2 = vor.points[ridge_points]
                    direction = np.array([p1[1] - p2[1], p2[0] - p1[0]])  # Perpendicular vector
                    direction = direction / np.linalg.norm(direction)  # Normalize
                    
                    # ✅ Compute intersection with boundary
                    max_x, max_y = 10, 10  # Arena boundary
                    t_max = min(
                        (max_x - finite_vertex[0]) / direction[0] if direction[0] > 0 else (-max_x - finite_vertex[0]) / direction[0],
                        (max_y - finite_vertex[1]) / direction[1] if direction[1] > 0 else (-max_y - finite_vertex[1]) / direction[1]
                    )

                    extended_vertex = finite_vertex + direction * t_max

                    edges.append([float(finite_vertex[0]), float(finite_vertex[1]),
                                float(extended_vertex[0]), float(extended_vertex[1])])

            self.get_logger().info(f"Found {len(edges)} valid Voronoi edges (including boundary extensions)")

            # Publish edges
            msg = String()
            msg.data = json.dumps(edges)
            self.voronoi_publisher.publish(msg)
        
        except Exception as e:
            self.get_logger().error(f"Voronoi computation failed: {e}")


    def is_inside_polygon(self, point, polygon):
        """ Check if a point is inside a polygon. """
        x, y = point
        poly_x, poly_y = polygon[:, 0], polygon[:, 1]
        return np.min(poly_x) <= x <= np.max(poly_x) and np.min(poly_y) <= y <= np.max(poly_y)

    def project_to_nearest_boundary(self, point, polygon):
        """ Move a point to the nearest point inside its Voronoi cell. """
        x, y = point
        distances = np.linalg.norm(polygon - np.array([x, y]), axis=1)
        return polygon[np.argmin(distances)]

def main():
    rclpy.init()
    node = VoronoiManager()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
