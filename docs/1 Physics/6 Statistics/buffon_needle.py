import numpy as np
import matplotlib.pyplot as plt
import time

def estimate_pi_buffon(n_needles, needle_length=1.0, line_spacing=2.0):
    """
    Estimate Pi using Buffon's Needle method.
    
    Args:
        n_needles: Number of needles to drop
        needle_length: Length of the needle
        line_spacing: Distance between parallel lines
        
    Returns:
        pi_estimate: Estimated value of Pi
        crossings: Number of needle crossings
        n_needles: Total number of needles
        execution_time: Time taken for the calculation
    """
    start_time = time.time()
    
    # Generate random positions and angles for needles
    # y: distance from the center of the needle to the nearest line (0 to line_spacing/2)
    # theta: angle of the needle (0 to pi)
    y = np.random.uniform(0, line_spacing/2, n_needles)
    theta = np.random.uniform(0, np.pi, n_needles)
    
    # Calculate if needle crosses a line
    # A needle crosses a line if y <= (needle_length/2) * sin(theta)
    crossings = np.sum(y <= (needle_length/2) * np.sin(theta))
    
    # Estimate Pi using Buffon's formula: pi = (2 * needle_length * n_needles) / (line_spacing * crossings)
    pi_estimate = (2 * needle_length * n_needles) / (line_spacing * crossings)
    
    execution_time = time.time() - start_time
    
    return pi_estimate, crossings, n_needles, execution_time

def plot_buffon_needle(n_needles, needle_length=1.0, line_spacing=2.0, save_path=None):
    """
    Visualize Buffon's Needle method for estimating Pi.
    
    Args:
        n_needles: Number of needles to drop
        needle_length: Length of the needle
        line_spacing: Distance between parallel lines
        save_path: Path to save the figure (optional)
    """
    # Generate random positions and angles for needles
    x = np.random.uniform(0, 10, n_needles)  # x-position (arbitrary)
    y = np.random.uniform(0, line_spacing, n_needles)  # y-position
    theta = np.random.uniform(0, np.pi, n_needles)  # angle
    
    # Calculate if needle crosses a line
    crossings = y <= (needle_length/2) * np.sin(theta)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot the parallel lines
    for i in range(0, 11, 2):
        plt.axhline(y=i*line_spacing, color='k', linestyle='-', alpha=0.5)
    
    # Plot the needles
    for i in range(n_needles):
        # Calculate endpoints of the needle
        dx = (needle_length/2) * np.cos(theta[i])
        dy = (needle_length/2) * np.sin(theta[i])
        
        # Plot the needle
        if crossings[i]:
            plt.plot([x[i]-dx, x[i]+dx], [y[i]-dy, y[i]+dy], 'r-', linewidth=1, alpha=0.7)
        else:
            plt.plot([x[i]-dx, x[i]+dx], [y[i]-dy, y[i]+dy], 'b-', linewidth=1, alpha=0.7)
    
    # Add labels and title
    plt.grid(True, alpha=0.3)
    plt.xlim(-1, 11)
    plt.ylim(-1, line_spacing*11)
    plt.title(f"Buffon's Needle: {n_needles} needles, {np.sum(crossings)} crossings")
    plt.xlabel("x")
    plt.ylabel("y")
    
    # Calculate and display the Pi estimate
    pi_estimate = (2 * needle_length * n_needles) / (line_spacing * np.sum(crossings))
    plt.text(0.05, -0.5, f'π ≈ {pi_estimate:.6f}', fontsize=12, 
             bbox=dict(facecolor='white', alpha=0.8), transform=plt.gca().transAxes)
    
    # Add legend
    plt.plot([], [], 'r-', label='Crossing')
    plt.plot([], [], 'b-', label='No Crossing')
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.tight_layout()
    plt.show()

def convergence_analysis(max_needles, step_size=1000, num_trials=5):
    """
    Analyze the convergence of the Pi estimation as the number of needles increases.
    
    Args:
        max_needles: Maximum number of needles to use
        step_size: Step size for increasing the number of needles
        num_trials: Number of trials for each needle count
        
    Returns:
        needle_counts: Array of needle counts used
        pi_estimates: Array of Pi estimates
        execution_times: Array of execution times
    """
    needle_counts = np.arange(step_size, max_needles + step_size, step_size)
    pi_estimates = np.zeros((len(needle_counts), num_trials))
    execution_times = np.zeros((len(needle_counts), num_trials))
    
    for i, n in enumerate(needle_counts):
        for j in range(num_trials):
            pi_est, _, _, exec_time = estimate_pi_buffon(n)
            pi_estimates[i, j] = pi_est
            execution_times[i, j] = exec_time
    
    return needle_counts, pi_estimates, execution_times

def plot_convergence(needle_counts, pi_estimates, save_path=None):
    """
    Plot the convergence of Pi estimates as the number of needles increases.
    
    Args:
        needle_counts: Array of needle counts used
        pi_estimates: Array of Pi estimates
        save_path: Path to save the figure (optional)
    """
    plt.figure(figsize=(12, 8))
    
    # Plot individual trials
    for j in range(pi_estimates.shape[1]):
        plt.plot(needle_counts, pi_estimates[:, j], 'o-', alpha=0.3, markersize=3)
    
    # Plot mean estimate
    mean_estimate = np.mean(pi_estimates, axis=1)
    plt.plot(needle_counts, mean_estimate, 'r-', linewidth=2, label='Mean Estimate')
    
    # Plot true value of Pi
    plt.axhline(y=np.pi, color='g', linestyle='--', label='True π')
    
    # Add error bands
    std_estimate = np.std(pi_estimates, axis=1)
    plt.fill_between(needle_counts, 
                     mean_estimate - std_estimate, 
                     mean_estimate + std_estimate, 
                     color='r', alpha=0.2, label='±1 Standard Deviation')
    
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    plt.xlabel('Number of Needles')
    plt.ylabel('Estimated π')
    plt.title('Convergence of Buffon\'s Needle π Estimation')
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.tight_layout()
    plt.show()

def compare_methods(max_points, step_size=1000, num_trials=3):
    """
    Compare the convergence of both Monte Carlo methods.
    
    Args:
        max_points: Maximum number of points/needles to use
        step_size: Step size for increasing the number of points/needles
        num_trials: Number of trials for each point/needle count
    """
    # Get convergence data for both methods
    point_counts, circle_estimates, circle_times = convergence_analysis(max_points, step_size, num_trials)
    needle_counts, needle_estimates, needle_times = convergence_analysis(max_points, step_size, num_trials)
    
    # Calculate mean estimates and standard deviations
    circle_mean = np.mean(circle_estimates, axis=1)
    circle_std = np.std(circle_estimates, axis=1)
    needle_mean = np.mean(needle_estimates, axis=1)
    needle_std = np.std(needle_estimates, axis=1)
    
    # Create the comparison plot
    plt.figure(figsize=(12, 8))
    
    # Plot circle method
    plt.plot(point_counts, circle_mean, 'b-', linewidth=2, label='Circle Method')
    plt.fill_between(point_counts, 
                     circle_mean - circle_std, 
                     circle_mean + circle_std, 
                     color='b', alpha=0.2, label='Circle ±1 Std Dev')
    
    # Plot needle method
    plt.plot(needle_counts, needle_mean, 'r-', linewidth=2, label='Buffon\'s Needle')
    plt.fill_between(needle_counts, 
                     needle_mean - needle_std, 
                     needle_mean + needle_std, 
                     color='r', alpha=0.2, label='Needle ±1 Std Dev')
    
    # Plot true value of Pi
    plt.axhline(y=np.pi, color='g', linestyle='--', label='True π')
    
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    plt.xlabel('Number of Points/Needles')
    plt.ylabel('Estimated π')
    plt.title('Comparison of Monte Carlo Methods for π Estimation')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("method_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot execution times
    plt.figure(figsize=(12, 8))
    
    circle_time_mean = np.mean(circle_times, axis=1)
    needle_time_mean = np.mean(needle_times, axis=1)
    
    plt.plot(point_counts, circle_time_mean, 'b-', linewidth=2, label='Circle Method')
    plt.plot(needle_counts, needle_time_mean, 'r-', linewidth=2, label='Buffon\'s Needle')
    
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.xlabel('Number of Points/Needles')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Computational Efficiency Comparison')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("efficiency_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Example usage
    n_needles = 10000
    pi_estimate, crossings, n_needles, exec_time = estimate_pi_buffon(n_needles)
    print(f"Estimated π: {pi_estimate:.6f}")
    print(f"Crossings: {crossings}/{n_needles}")
    print(f"Execution time: {exec_time:.4f} seconds")
    
    # Visualize the method
    plot_buffon_needle(100, save_path="buffon_needle.png")
    
    # Analyze convergence
    needle_counts, pi_estimates, exec_times = convergence_analysis(100000, step_size=10000, num_trials=3)
    plot_convergence(needle_counts, pi_estimates, save_path="buffon_convergence.png")
    
    # Compare methods
    compare_methods(100000, step_size=10000, num_trials=3) 