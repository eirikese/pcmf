import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

# Function to calculate the 80% confidence interval for a distribution
def confidence_interval(data, dist, params, confidence=0.8):
    mean, var = dist.stats(*params, moments='mv')
    sigma = np.sqrt(var)
    h = sigma * stats.norm.ppf((1 + confidence) / 2)
    return mean - h, mean + h

# Function to compute AIC for a distribution
def calculate_aic(n, ll, k):
    return 2 * k - 2 * ll

# Read the CSV file into a DataFrame
data = pd.read_csv('pcmf/plot_scripts/eval_data.csv')  # Replace with the actual path to your CSV file

# Set up the subplots
fig, ax = plt.subplots(3, 2, figsize=(18, 15))

# List of distributions to test
distributions = [stats.norm, stats.lognorm, stats.expon, stats.gamma]

# Plots for transformation distributions and fit different distributions
for i, col in enumerate(['transformation_history_x', 'transformation_history_y', 'transformation_history_rotation']):
    data_col = data[col].dropna().to_numpy()  # Clean the data
    xmin, xmax = min(data_col), max(data_col)
    x = np.linspace(xmin, xmax, 100)
    
    # Fit and plot each distribution
    for dist in distributions:
        # Fit distribution
        params = dist.fit(data_col)
        # Calculate fitted PDF and error from fit in the data
        pdf = dist.pdf(x, *params)
        
        # Plot the PDF
        ax[i, 1].plot(x, pdf, label=f'{dist.name} fit')

        # Calculate and plot the 80% confidence interval
        low, high = confidence_interval(x, dist, params)
        ax[i, 1].fill_between(x, pdf, where=(low < x) & (x < high), alpha=0.2)
        
        # Compute AIC and print
        ll = np.sum(np.log(dist.pdf(data_col, *params)))
        aic = calculate_aic(len(data_col), ll, len(params))
        print(f'AIC for {dist.name} on {col}: {aic}')

    # Plot histogram for the data
    ax[i, 1].hist(data_col, bins=30, density=True, alpha=0.3, color='gray')
    
    # Set labels and titles
    ax[i, 1].set_xlabel(['X Position', 'Y Position', 'Theta'][i])
    ax[i, 1].set_ylabel('Probability Density')
    ax[i, 1].legend()
    ax[i, 1].grid(True)
    ax[i, 1].set_xlim(xmin, xmax)

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plot
plt.show()
