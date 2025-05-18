# Re-run the script since execution state was reset

import json
import matplotlib.pyplot as plt
import numpy as np
import os

# Define log files and their respective conditions
log_files = {
    "No Voronoi-based Coordination": "min_distance_log_1.json",
    "Buffered Voronoi Cells (BVC) Only": "min_distance_log_2.json",
    "Intruder Introduced": "min_distance_log_3.json",
    "HJ Reachability Activated": "min_distance_log_4.json"
}

# Define colors for each condition
colors = {
    "No Voronoi-based Coordination": "red",
    "Buffered Voronoi Cells (BVC) Only": "blue",
    "Intruder Introduced": "green",
    "HJ Reachability Activated": "purple"
}

plt.figure(figsize=(10, 5))

# Iterate through each log file and plot its data
for condition, log_file in log_files.items():
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            data = json.load(f)

        if data:
            timestamps = np.array([entry["time"] for entry in data])
            min_distances = np.array([entry["min_distance"] for entry in data])

            # Define the time window (last 2 minutes)
            latest_time = timestamps[-1]
            time_window_start = latest_time - 120  # Last 120 seconds

            # Filter data within the last 2 minutes
            valid_indices = timestamps >= time_window_start
            timestamps = timestamps[valid_indices]
            min_distances = min_distances[valid_indices]

            # Convert timestamps to relative time (seconds from start of window)
            timestamps = timestamps - timestamps[0]

            # Plot the data with a thin line
            plt.plot(
                timestamps,
                min_distances,
                linestyle="-",
                linewidth=1,  # Thin line
                color=colors[condition],
                label=condition
            )

# Plot collision threshold line
plt.axhline(y=0.5, color='black', linestyle='--', linewidth=1, label="Collision Threshold (0.5m)")

# Configure plot labels and legend
plt.xlabel("Time (seconds)")
plt.ylabel("Minimum Distance (m)")
plt.title("Minimum Distance Between Robots (Last 2 Minutes)")
plt.legend()
plt.grid(True)
plt.show()

# Re-run the script with larger text and thicker lines

plt.figure(figsize=(10, 5))

# Increase font size globally
plt.rcParams.update({'font.size': 18})  

# Iterate through each log file and plot its data
for condition, log_file in log_files.items():
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            data = json.load(f)

        if data:
            timestamps = np.array([entry["time"] for entry in data])
            min_distances = np.array([entry["min_distance"] for entry in data])

            # Define the time window (last 2 minutes)
            latest_time = timestamps[-1]
            time_window_start = latest_time - 120  # Last 120 seconds

            # Filter data within the last 2 minutes
            valid_indices = timestamps >= time_window_start
            timestamps = timestamps[valid_indices]
            min_distances = min_distances[valid_indices]

            # Convert timestamps to relative time (seconds from start of window)
            timestamps = timestamps - timestamps[0]

            # Plot the data with thicker lines
            plt.plot(
                timestamps,
                min_distances,
                linestyle="-",
                linewidth=3,  # Thicker lines for better visibility
                color=colors[condition],
                label=condition
            )

# Plot collision threshold line with thicker width
plt.axhline(y=0.5, color='black', linestyle='--', linewidth=3, label="Collision Threshold (0.5m)")

# Configure plot labels and legend with larger font
plt.xlabel("Time (seconds)", fontsize=20)
plt.ylabel("Minimum Distance (m)", fontsize=20)
plt.title("Minimum Distance Between Robots (Last 2 Minutes)", fontsize=22)
plt.legend(fontsize=16)
plt.grid(True, linewidth=1.5)

# Show the improved plot
plt.show()

