import numpy as np
import rclpy
from rclpy.node import Node
from scipy.spatial import Voronoi
from geometry_msgs.msg import Pose
from std_msgs.msg import String
import json

# Shapely imports
from shapely.geometry import Polygon, box

class VoronoiManager(Node):
    def __init__(self, excluded_robot_id=3):  # ⬅️ Add excluded robot ID
        super().__init__('voronoi_manager')
        self.excluded_robot_id = excluded_robot_id 

        # Single consistent set of boundary points
        self.boundary_points = np.array([
            [-12, -12], [  0, -12], [ 12, -12],
            [-12,   0], [ 12,   0],
            [-12,  12], [  0,  12], [ 12,  12]
        ])
        self.safety_radius = 0.3
        self.robot_positions = {}
        self.robot_voronoi_indices = {}

        ######
        # We'll track up to 6 robots, each with its own BVC publisher
        self.num_robots = 6
        self.bvc_publishers = {}
        ######

        for robot_id in range(1, self.num_robots+1):
            if robot_id == self.excluded_robot_id:  # ⬅️ Skip this robot
                self.get_logger().info(f"🚫 Robot {robot_id} is excluded from Voronoi computation.")
                continue  
            self.create_subscription(
                Pose,
                f'/robot_{robot_id}/pose',
                lambda msg, rid=robot_id: self.update_position(msg, rid),
                10
            )
            # Create a dedicated publisher for each robot’s BVC
            pub = self.create_publisher(String, f'/robot_{robot_id}/bvc_edges', 10)
            self.bvc_publishers[robot_id] = pub

        # Publishers
        self.voronoi_publisher = self.create_publisher(String, '/voronoi_edges', 10)
        self.bvc_publisher = self.create_publisher(String, '/buffered_voronoi_edges', 10)

        # Timer to compute & publish Voronoi
        self.create_timer(0.5, self.compute_and_publish_voronoi)
        self.get_logger().info("VoronoiManager node started.")

    def update_position(self, msg, robot_id):
        """Store the (x, y) positions of each robot."""
        self.robot_positions[robot_id] = (msg.position.x, msg.position.y)

    def compute_and_publish_voronoi(self):
        """Compute Voronoi & Buffered Voronoi and publish them."""
        if len(self.robot_positions) < 2:
            self.get_logger().info("Not enough robot positions for Voronoi.")
            return

        # Sort the robot IDs for a stable ordering
        robot_ids_sorted = sorted(
            rid for rid in self.robot_positions.keys() if rid != self.excluded_robot_id  # ⬅️ Remove excluded robot
        )
        current_positions = [self.robot_positions[rid] for rid in robot_ids_sorted]


        all_points = np.vstack([current_positions, self.boundary_points])

        # Build Voronoi
        try:
            vor = Voronoi(all_points)
        except Exception as e:
            self.get_logger().error(f"Voronoi construction failed: {e}")
            return

        for i, rid in enumerate(robot_ids_sorted):
            self.robot_voronoi_indices[rid] = i

        # Accumulate edges
        voronoi_edges = []
        all_bvc_edges = []
        '''
        for i, rid in enumerate(robot_ids_sorted):
            # 1) Standard Voronoi region (finite, clipped).
            edges = self.extract_clipped_cell(vor, i)
            voronoi_edges.extend(edges)

            # 2) Buffered Voronoi region (also clipped), i.e. "shrunk" by safety_radius.
            edges_bvc = self.extract_buffered_cell(vor, i)
            all_bvc_edges.extend(edges_bvc)
        '''
        for i, rid in enumerate(robot_ids_sorted):
            edges_vor = self.extract_clipped_cell(vor, i)
            voronoi_edges.extend(edges_vor)

            edges_bvc = self.extract_buffered_cell(vor, i)
            all_bvc_edges.extend(edges_bvc)

            if rid == self.excluded_robot_id:  # ⬅️ Do NOT publish BVC for excluded robot
                continue

            # Publish only this robot's BVC edges to its dedicated topic
            msg_bvc = String()
            msg_bvc.data = json.dumps(edges_bvc)
            self.bvc_publishers[rid].publish(msg_bvc)




        self.get_logger().info(f"Voronoi edges: {len(voronoi_edges)}; BVC edges: {len(all_bvc_edges)}")

        # Publish standard Voronoi Edges
        msg_vor = String()
        msg_vor.data = json.dumps(voronoi_edges)
        self.voronoi_publisher.publish(msg_vor)

        # Publish BVC Edges
        msg_bvc = String()
        msg_bvc.data = json.dumps(all_bvc_edges)
        self.bvc_publisher.publish(msg_bvc)


    def extract_clipped_cell(self, vor, robot_index):
        """
        Build the Voronoi cell polygon, clip it to [-10,10]x[-10,10], 
        and return edges as (x1,y1,x2,y2).
        """
        edges = []

        bounding_poly = box(-10, -10, 10, 10)
        region_index = vor.point_region[robot_index]
        region = vor.regions[region_index]

        if not region or all(v == -1 for v in region):
            return edges

        # Gather polygon points, ignoring -1 (infinite edges)
        pts = []
        for v in region:
            if v != -1:
                pts.append(vor.vertices[v])
        if len(pts) < 3:
            return edges

        poly = Polygon(pts)
        clipped_poly = bounding_poly.intersection(poly)
        # Convert the clipped polygon into edges:
        edges.extend(self.polygon_to_edges(clipped_poly))

        return edges


    def extract_buffered_cell(self, vor, robot_index):
        """
        Similar to extract_clipped_cell, but we also
        'shrink' (inward offset) the polygon by safety_radius 
        before extracting edges.
        """
        edges = []

        bounding_poly = box(-10, -10, 10, 10)
        region_index = vor.point_region[robot_index]
        region = vor.regions[region_index]

        if not region or all(v == -1 for v in region):
            return edges

        pts = []
        for v in region:
            if v != -1:
                pts.append(vor.vertices[v])
        if len(pts) < 3:
            return edges

        poly = Polygon(pts)
        clipped_poly = bounding_poly.intersection(poly)

        # [Key difference] Now we offset inward using buffer(-radius)
        # Negative buffer means "shrink" the polygon
        # You can adjust resolution if you want more precision
        bvc_poly = clipped_poly.buffer(-self.safety_radius)

        edges.extend(self.polygon_to_edges(bvc_poly))
        return edges


    def polygon_to_edges(self, shapely_geom):
        """
        Convert a Shapely Polygon or MultiPolygon into a list of edges
        [x1, y1, x2, y2]. Handles holes and multipolygons.
        """
        results = []
        if shapely_geom.is_empty:
            return results

        # Could be polygon or multi-polygon
        if shapely_geom.geom_type == "MultiPolygon":
            geoms = shapely_geom.geoms
        else:
            geoms = [shapely_geom]

        for g in geoms:
            # Exterior
            exterior_coords = list(g.exterior.coords)
            for i in range(len(exterior_coords) - 1):
                x1, y1 = exterior_coords[i]
                x2, y2 = exterior_coords[i+1]
                results.append([float(x1), float(y1), float(x2), float(y2)])
            
            # Interiors (holes)
            for interior in g.interiors:
                interior_coords = list(interior.coords)
                for i in range(len(interior_coords) - 1):
                    x1, y1 = interior_coords[i]
                    x2, y2 = interior_coords[i+1]
                    results.append([float(x1), float(y1), float(x2), float(y2)])
        return results


def main():
    rclpy.init()
    node = VoronoiManager(excluded_robot_id=3)  # ⬅️ Exclude robot 3
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
