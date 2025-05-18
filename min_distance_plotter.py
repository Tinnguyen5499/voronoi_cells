# Re-run the script since execution state was reset

import json
import matplotlib.pyplot as plt
import numpy as np
import os

# Define the log file path
log_file = "min_distance_log.json"

# Check if the file exists
if os.path.exists(log_file):
    # Read data from the log file
    with open(log_file, 'r') as f:
        data = json.load(f)

    # Convert data to numpy arrays
    timestamps = np.array([entry["time"] for entry in data])
    min_distances = np.array([entry["min_distance"] for entry in data])

    # Define the time window (last 2 minutes)
    if len(timestamps) > 0:
        latest_time = timestamps[-1]
        time_window_start = latest_time - 120  # Last 120 seconds

        # Filter data within the last 2 minutes
        valid_indices = timestamps >= time_window_start
        timestamps = timestamps[valid_indices]
        min_distances = min_distances[valid_indices]

        # Convert timestamps to relative time (seconds from start of window)
        timestamps = timestamps - timestamps[0]

        # Plot the data
        plt.figure(figsize=(10, 5))
        plt.plot(timestamps, min_distances, marker='o', linestyle='-', color='b', label="Minimum Distance")
        plt.axhline(y=0.3, color='r', linestyle='--', label="Collision Threshold (0.5m)")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Minimum Distance (m)")
        plt.title("Minimum Distance Between Robots (Last 2 Minutes)")
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        print("No data available in the log file.")
else:
    print(f"Log file '{log_file}' not found. Make sure the distance monitoring script has generated data.")

