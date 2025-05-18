import rclpy
from rclpy.executors import MultiThreadedExecutor
from simulation import Simulation
from visualization import Visualization
from goal_manager import GoalManager
import threading
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

# ✅ Ensure Matplotlib uses a GUI-compatible backend
matplotlib.use("TkAgg")  # Change to "Qt5Agg" if TkAgg fails

def main():
    rclpy.init()

    # Define robot IDs and their initial positions
    robot_configs = [
        {"id": 1, "x": 3.0, "y": 5.0, "theta": 360*np.pi/180},
        {"id": 2, "x": -3.0, "y": 0.0, "theta": 360*np.pi/180},
        {"id": 3, "x": 2.0, "y": 0.0, "theta": 180*np.pi/180},
        #{"id": 4, "x": -3.0, "y": 0.0, "theta": 360*np.pi/180}, 
        #{"id": 5, "x": -1.5, "y": -2.6, "theta": 420*np.pi/180}, 
        #{"id": 6, "x": 1.5, "y": -2.5, "theta": 480*np.pi/180}, 
    ]

    # Extract robot IDs
    robot_ids = [config["id"] for config in robot_configs]

    # ✅ Create instances of Simulation and Visualization
    sim = Simulation(dt=0.1, robot_ids=robot_ids)
    vis = Visualization(robot_ids=robot_ids)  # Visualization node

    # ✅ Set initial positions for robots
    for config in robot_configs:
        sim.robots[config["id"]].x = config["x"]
        sim.robots[config["id"]].y = config["y"]
        sim.robots[config["id"]].theta = config["theta"]
    '''
    # ✅ Set goals WITH heading (theta in radians)
    sim.set_goal(1, (7.5, 7, np.radians(90)))    
    sim.set_goal(2, (4.5, 7, np.radians(90)))  
    sim.set_goal(3, (1.5, 7, np.radians(90)))   
    sim.set_goal(4, (-1.5, 7, np.radians(90)))    
    sim.set_goal(5, (-4.5, 7, np.radians(90)))  
    sim.set_goal(6, (-7.5, 7, np.radians(90)))  
    '''
    # ✅ MultiThreadedExecutor to spin all nodes concurrently
    executor = MultiThreadedExecutor()
    executor.add_node(sim)
    executor.add_node(vis)  
    executor.add_node(sim.goal_manager)

    # ✅ Run ROS 2 in a separate thread
    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()

    # ✅ Run the simulation loop in a separate thread
    sim_thread = threading.Thread(target=sim.run, daemon=True)
    sim_thread.start()

    # ✅ Store animation globally to prevent garbage collection
    def update_visualization(_):
        vis.update_plot()
        vis.fig.canvas.flush_events()  # Ensure Matplotlib processes updates

    global ani  # ✅ Store globally
    ani = animation.FuncAnimation(vis.fig, update_visualization, interval=100)

    print("🚀 Simulation & Visualization Running...")

    try:
        while True:  # ✅ This prevents the script from exiting
            plt.pause(1)  # ✅ Keep Matplotlib interactive
    except KeyboardInterrupt:
        print("🛑 Shutting down ROS nodes...")
        rclpy.shutdown()
        executor_thread.join()
        sim_thread.join()

        # ✅ Destroy nodes properly
        sim.destroy_node()
        vis.destroy_node()
        sim.goal_manager.destroy_node()

if __name__ == '__main__':
    main()
