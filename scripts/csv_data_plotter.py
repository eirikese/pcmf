import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data from a CSV file
file_path = 'pcmf/pcmf_data/pcmf_data_1709281519.csv'
data = pd.read_csv(file_path)

# Generate a range of integers based on the length of the data to use as the x-axis
x_values = np.arange(len(data))

# Extracting data columns
processing_times = data['Processing Time'].to_numpy()
number_of_corners_detected = data['Number of Corners Detected'].to_numpy()
anchor_rmse = data['Anchor RMSE'].to_numpy()
gicp_rmse = data['GICP RMSE'].to_numpy()

# Begin plotting
plt.figure(figsize=(20, 10))

# Plot Processing Time
plt.subplot(2, 2, 1)
plt.plot(x_values, processing_times, label='Processing Time')
plt.xlabel('Data Point')
plt.ylabel('Processing Time (ms)')
plt.title('Processing Time over Data Points')
plt.legend()

# Plot Number of Corners Detected
plt.subplot(2, 2, 2)
plt.plot(x_values, number_of_corners_detected, label='Number of Corners Detected', color='orange')
plt.xlabel('Data Point')
plt.ylabel('Number of Corners Detected')
plt.title('Number of Corners Detected over Data Points')
plt.legend()

# Plot Anchor RMSE
plt.subplot(2, 2, 3)
plt.plot(x_values, anchor_rmse, label='Anchor RMSE', color='green')
plt.xlabel('Data Point')
plt.ylabel('Anchor RMSE (units)')
plt.title('Anchor RMSE over Data Points')
plt.legend()

# Plot GICP RMSE
plt.subplot(2, 2, 4)
plt.plot(x_values, gicp_rmse, label='GICP RMSE', color='red')
plt.xlabel('Data Point')
plt.ylabel('GICP RMSE (units)')
plt.title('GICP RMSE over Data Points')
plt.legend()

# Adjust layout and display the plot
plt.tight_layout()
plt.show()
