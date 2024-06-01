import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

# Function to calculate the 80% confidence interval for normal distribution
def confidence_interval(data, confidence=0.8):
    mean, sigma = np.mean(data), np.std(data)
    h = sigma * norm.ppf((1 + confidence) / 2)
    return mean - h, mean + h

# Read the CSV file into a DataFrame
data = pd.read_csv('pcmf/plot_scripts/eval_data.csv')  # Replace with the actual path to your CSV file

# Set up the subplots
fig, ax = plt.subplots(3, 2, figsize=(18, 15))

# Plots for transformations over time
ax[0, 0].plot(data['timestamp'].to_numpy(), data['transformation_history_x'].to_numpy(), label='X without EKF', color='orange')
ax[0, 0].plot(data['timestamp'].to_numpy(), data['EKF_transformation_history_x'].to_numpy(), label='X with EKF', color='green')
ax[1, 0].plot(data['timestamp'].to_numpy(), data['transformation_history_y'].to_numpy(), label='Y without EKF', color='orange')
ax[1, 0].plot(data['timestamp'].to_numpy(), data['EKF_transformation_history_y'].to_numpy(), label='Y with EKF', color='green')
ax[2, 0].plot(data['timestamp'].to_numpy(), data['transformation_history_rotation'].to_numpy(), label='Theta without EKF', color='orange')
ax[2, 0].plot(data['timestamp'].to_numpy(), data['EKF_transformation_history_rotation'].to_numpy(), label='Theta with EKF', color='green')

# Set labels and titles for left side plots
for i in range(3):
    ax[i, 0].set_xlabel('Timestamp')
    ax[i, 0].set_ylabel(['X Position', 'Y Position', 'Theta'][i])
    ax[i, 0].legend()
    ax[i, 0].grid(True)

# Plots for transformation distributions
for i, col in enumerate(['transformation_history_x', 'transformation_history_y', 'transformation_history_rotation']):
    data_col = data[col].dropna().to_numpy()
    ekf_col = data[f'EKF_{col}'].dropna().to_numpy()

    # Calculate normal distribution fit
    mu, std = norm.fit(data_col)
    mu_ekf, std_ekf = norm.fit(ekf_col)

    # Create a dense range of x values for plotting the PDF
    xmin, xmax = min(data_col.min(), ekf_col.min()), max(data_col.max(), ekf_col.max())
    x = np.linspace(xmin, xmax, 1000)
    p = norm.pdf(x, mu, std)
    p_ekf = norm.pdf(x, mu_ekf, std_ekf)

    # Manually normalize the PDF so that the integral is exactly 1
    p /= np.trapz(p, x)
    p_ekf /= np.trapz(p_ekf, x)

    # Plot the distributions
    ax[i, 1].plot(x, p, label=f'{col} Normalized', color='orange')
    ax[i, 1].plot(x, p_ekf, label=f'EKF_{col} Normalized', color='green')

    # Plotting the 80% confidence interval shading
    low, high = confidence_interval(data_col, confidence=0.8)
    low_ekf, high_ekf = confidence_interval(ekf_col, confidence=0.8)
    ax[i, 1].fill_between(x, p, where=(x > low) & (x < high), color='orange', alpha=0.2)
    ax[i, 1].fill_between(x, p_ekf, where=(x > low_ekf) & (x < high_ekf), color='green', alpha=0.2)

    # Set labels and grid
    ax[i, 1].set_xlabel(['X Position', 'Y Position', 'Theta'][i])
    ax[i, 1].set_ylabel('Probability Density')
    ax[i, 1].legend()
    ax[i, 1].grid(True)


# Show the plot
plt.show()
