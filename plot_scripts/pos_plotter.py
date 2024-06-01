# Rewrite the plotting script considering the issues mentioned earlier
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file into a DataFrame
data = pd.read_csv('pcmf/plot_scripts/eval_data.csv')  # Replace with the actual path to your CSV file

# Plot the local fix and path points
fig, ax = plt.subplots(3, 1, figsize=(12, 18))

# # limits for the plot
# ax[0].set_xlim(-150, -100)
# ax[0].set_ylim(-50, 50)

# First subplot for the paths in xy plane
# ax[0].plot(data['gps/local_fix_x'].to_numpy(), data['gps/local_fix_y'].to_numpy(), label='GPS Local Fix Path', linewidth=2)
# ax[0].plot(data['path_points_x'].to_numpy(), data['path_points_y'].to_numpy(), label='Path Points', linewidth=2)
ax[0].scatter(data['gps/local_fix_x'].to_numpy(), data['gps/local_fix_y'].to_numpy(), label='GPS Path', color='lightblue', s = 5)
ax[0].scatter(data['path_points_x'].to_numpy(), data['path_points_y'].to_numpy(), label='PCMF path points', color='orange', s = 5)
ax[0].set_xlabel('X Position')
ax[0].set_ylabel('Y Position')
ax[0].set_title('Paths in XY plane')
ax[0].legend()
ax[0].grid(True)
# equal aspect ratio and scaling
ax[0].set_aspect('equal', 'box')

# Draw error lines between the points
# for i in range(len(data)):
#     ax[0].plot([data['gps/local_fix_x'].iloc[i], data['path_points_x'].iloc[i]], 
#                [data['gps/local_fix_y'].iloc[i], data['path_points_y'].iloc[i]], 
#                'lightgrey', linestyle='dotted')

# Second subplot for error over time
errors = np.sqrt((data['gps/local_fix_x'] - data['path_points_x'])**2 + (data['gps/local_fix_y'] - data['path_points_y'])**2)
ax[1].plot(data['timestamp'].to_numpy(), errors.to_numpy(), label='Error Over Time', color='red')
ax[1].set_xlabel('Timestamp')
ax[1].set_ylabel('Error')
ax[1].set_title('Error Over Time')
ax[1].grid(True)
ax[1].legend()

# Third subplot for boxplot of errors
ax[2].boxplot(errors, vert=False, patch_artist=True)
ax[2].set_title('Boxplot of Errors')
ax[2].set_xlabel('Error')
ax[2].grid(True)

# Adjust layout to prevent overlap
plt.tight_layout(h_pad=5.0, pad=3.0, w_pad=3.0)

# Show the plot
plt.show()