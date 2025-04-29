import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import os

def analyze_pendulum_data(length, length_uncertainty, time_measurements):
    """
    Analyze pendulum data to determine gravitational acceleration.
    
    Parameters:
    -----------
    length : float
        Length of the pendulum in meters
    length_uncertainty : float
        Uncertainty in the length measurement in meters
    time_measurements : list or array
        List of time measurements for 10 oscillations in seconds
        
    Returns:
    --------
    dict
        Dictionary containing analysis results
    """
    # Convert to numpy array for easier manipulation
    times = np.array(time_measurements)
    
    # Calculate statistics for 10 oscillations
    mean_time_10 = np.mean(times)
    std_time_10 = np.std(times, ddof=1)  # ddof=1 for sample standard deviation
    n = len(times)
    uncertainty_time_10 = std_time_10 / np.sqrt(n)
    
    # Calculate period and its uncertainty
    period = mean_time_10 / 10
    period_uncertainty = uncertainty_time_10 / 10
    
    # Calculate gravitational acceleration
    g = 4 * np.pi**2 * length / period**2
    
    # Calculate uncertainty in g using error propagation
    relative_length_uncertainty = length_uncertainty / length
    relative_period_uncertainty = period_uncertainty / period
    g_uncertainty = g * np.sqrt(relative_length_uncertainty**2 + (2 * relative_period_uncertainty)**2)
    
    # Calculate percent error compared to accepted value (9.81 m/s²)
    accepted_g = 9.81
    percent_error = abs(g - accepted_g) / accepted_g * 100
    
    # Return results
    results = {
        'length': length,
        'length_uncertainty': length_uncertainty,
        'mean_time_10': mean_time_10,
        'std_time_10': std_time_10,
        'uncertainty_time_10': uncertainty_time_10,
        'period': period,
        'period_uncertainty': period_uncertainty,
        'g': g,
        'g_uncertainty': g_uncertainty,
        'percent_error': percent_error,
        'raw_times': times
    }
    
    return results

def plot_time_measurements(times, save_path=None):
    """
    Plot the time measurements for 10 oscillations.
    
    Parameters:
    -----------
    times : list or array
        List of time measurements for 10 oscillations in seconds
    save_path : str, optional
        Path to save the figure
    """
    plt.figure(figsize=(10, 6))
    
    # Plot individual measurements
    plt.plot(range(1, len(times) + 1), times, 'bo-', label='Measurements')
    
    # Plot mean
    mean_time = np.mean(times)
    plt.axhline(y=mean_time, color='r', linestyle='--', label=f'Mean: {mean_time:.3f} s')
    
    # Add error bars for standard deviation
    std_time = np.std(times, ddof=1)
    plt.fill_between([1, len(times)], 
                     mean_time - std_time, 
                     mean_time + std_time, 
                     color='r', alpha=0.2, 
                     label=f'Standard Deviation: ±{std_time:.3f} s')
    
    plt.xlabel('Measurement Number')
    plt.ylabel('Time for 10 Oscillations (s)')
    plt.title('Pendulum Time Measurements')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    plt.show()
    plt.close()

def plot_histogram(times, save_path=None):
    """
    Plot a histogram of the time measurements.
    
    Parameters:
    -----------
    times : list or array
        List of time measurements for 10 oscillations in seconds
    save_path : str, optional
        Path to save the figure
    """
    plt.figure(figsize=(10, 6))
    
    # Plot histogram
    n, bins, patches = plt.hist(times, bins=5, alpha=0.7, color='skyblue', edgecolor='black')
    
    # Add a normal distribution curve
    mean_time = np.mean(times)
    std_time = np.std(times, ddof=1)
    x = np.linspace(min(times) - 0.5, max(times) + 0.5, 100)
    y = stats.norm.pdf(x, mean_time, std_time) * len(times) * (bins[1] - bins[0])
    plt.plot(x, y, 'r-', linewidth=2, label=f'Normal Distribution\nμ = {mean_time:.3f} s, σ = {std_time:.3f} s')
    
    plt.xlabel('Time for 10 Oscillations (s)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Pendulum Time Measurements')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    plt.show()
    plt.close()

def plot_length_vs_period(lengths, periods, period_uncertainties, save_path=None):
    """
    Plot the relationship between pendulum length and period.
    
    Parameters:
    -----------
    lengths : list or array
        List of pendulum lengths in meters
    periods : list or array
        List of measured periods in seconds
    period_uncertainties : list or array
        List of period uncertainties in seconds
    save_path : str, optional
        Path to save the figure
    """
    plt.figure(figsize=(10, 6))
    
    # Plot data points with error bars
    plt.errorbar(lengths, periods, yerr=period_uncertainties, fmt='o', capsize=5, 
                label='Measurements', color='blue')
    
    # Fit a power law (T = 2π√(L/g))
    # Linearize the data: T² = 4π²L/g
    lengths_squared = np.array(lengths)
    periods_squared = np.array(periods)**2
    
    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(lengths_squared, periods_squared)
    
    # Calculate g from the slope: slope = 4π²/g
    g_fit = 4 * np.pi**2 / slope
    
    # Plot the fit
    x_fit = np.linspace(min(lengths), max(lengths), 100)
    y_fit = np.sqrt(slope * x_fit + intercept)
    plt.plot(x_fit, y_fit, 'r-', label=f'Fit: T = {np.sqrt(slope):.2f}√L\n g = {g_fit:.2f} m/s²')
    
    plt.xlabel('Pendulum Length (m)')
    plt.ylabel('Period (s)')
    plt.title('Pendulum Length vs Period')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    plt.show()
    plt.close()
    
    return g_fit

def print_results(results):
    """
    Print the analysis results in a formatted way.
    
    Parameters:
    -----------
    results : dict
        Dictionary containing analysis results
    """
    print("===== PENDULUM ANALYSIS RESULTS =====")
    print(f"Pendulum Length: {results['length']:.3f} ± {results['length_uncertainty']:.3f} m")
    print(f"Mean Time for 10 Oscillations: {results['mean_time_10']:.3f} ± {results['uncertainty_time_10']:.3f} s")
    print(f"Standard Deviation: {results['std_time_10']:.3f} s")
    print(f"Period: {results['period']:.3f} ± {results['period_uncertainty']:.3f} s")
    print(f"Gravitational Acceleration: {results['g']:.3f} ± {results['g_uncertainty']:.3f} m/s²")
    print(f"Percent Error: {results['percent_error']:.2f}%")
    print("=====================================")

def main():
    """
    Main function to run the pendulum analysis and generate visualizations.
    """
    print("Analyzing pendulum data and generating visualizations...")
    
    # Create output directory if it doesn't exist
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Example data (replace with your actual measurements)
    length = 1.0  # meters
    length_uncertainty = 0.005  # meters (half of 1 cm resolution)
    
    # Example time measurements for 10 oscillations (in seconds)
    time_measurements = [
        20.1, 20.3, 20.0, 20.2, 20.1,
        20.4, 20.0, 20.2, 20.1, 20.3
    ]
    
    # Analyze the data
    results = analyze_pendulum_data(length, length_uncertainty, time_measurements)
    
    # Print results
    print_results(results)
    
    # Create visualizations
    print("Generating time measurements plot...")
    plot_time_measurements(results['raw_times'], save_path=os.path.join(output_dir, "pendulum_time_measurements.png"))
    
    print("Generating histogram plot...")
    plot_histogram(results['raw_times'], save_path=os.path.join(output_dir, "pendulum_histogram.png"))
    
    # Example data for multiple lengths
    print("Generating length vs period plot...")
    lengths = [0.5, 0.75, 1.0, 1.25, 1.5]
    periods = [1.42, 1.74, 2.01, 2.24, 2.46]
    period_uncertainties = [0.02, 0.02, 0.02, 0.02, 0.02]
    
    g_fit = plot_length_vs_period(lengths, periods, period_uncertainties, save_path=os.path.join(output_dir, "length_vs_period.png"))
    print(f"Gravitational acceleration from fit: {g_fit:.3f} m/s²")
    
    print(f"All visualizations have been saved to the '{output_dir}' directory.")

if __name__ == "__main__":
    main() 