import matplotlib.pyplot as plt
import pandas as pd

# Function to read float values from a text file
def read_float_values(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        float_values = [float(line.strip()) for line in lines]
    return float_values

# Function to plot float values with rolling average
def plot_float_values(float_values, window_size=100):
    # Calculate rolling average
    rolling_average = pd.Series(float_values).rolling(window=window_size, min_periods=1).mean()

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(float_values, linestyle='-', color='b', alpha=0.7, label='Float Values')
    plt.plot(rolling_average, linestyle='-', color='r', label=f'Rolling Average (window={window_size})')
    plt.title('Plot of Float Values with Rolling Average')
    plt.xlabel('Index')
    plt.ylabel('Float Value')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Main function
def main():
    # Example filename: discriminator_log_2.txt
    filename = 'temp/generator_log.txt'  # Replace with your file name
    float_values = read_float_values(filename)
    plot_float_values(float_values)

if __name__ == '__main__':
    main()