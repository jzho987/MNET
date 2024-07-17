import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Function to read float values from a text file
def read_float_values(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        float_values = [float(line.strip()) for line in lines]
    return float_values

# Function to plot float values with rolling average
def plot_float_values(float_values1, float_values2, window_size=170):
    # Calculate rolling average
    rolling_average1 = pd.Series(float_values1).rolling(window=window_size, min_periods=1).mean()
    rolling_average2 = pd.Series(float_values2).rolling(window=window_size, min_periods=1).mean()

    # Divide x-axis values by 170
    max_len = min(len(float_values1), len(float_values2))
    x_values = np.arange(max_len) / 170.0  # Adjust x-axis values

    rolling_average1 = rolling_average1[:max_len]
    rolling_average2 = rolling_average2[:max_len]

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, rolling_average1, linestyle='-', color='r', label=f'Generator')
    plt.plot(x_values, rolling_average2, linestyle='-', color='b', alpha=0.7, label='Discriminator')
    plt.title('Plot of Loss Values per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Main function
def main():
    # Example filename: discriminator_log_2.txt
    filename1 = 'temp/generator_log.txt'
    filename2 = 'temp/discriminator_log.txt'
    float_values1 = read_float_values(filename1)
    float_values2 = read_float_values(filename2)
    plot_float_values(float_values1, float_values2)

if __name__ == '__main__':
    main()