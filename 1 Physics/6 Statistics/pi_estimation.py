import numpy as np
import matplotlib.pyplot as plt
import time
import os

# ===== CIRCLE METHOD =====

def estimate_pi_circle(n_points=10000):
    """
    Estimate Pi using the circle method (Monte Carlo).
    
    Args:
        n_points: Number of random points to generate
        
    Returns:
        Estimated value of Pi
    """
    # Generate random points in the square [-1, 1] x [-1, 1]
    x = np.random.uniform(-1, 1, n_points)
    y = np.random.uniform(-1, 1, n_points)
    
    # Calculate distance from origin
    distance = np.sqrt(x**2 + y**2)
    
    # Count points inside the circle (radius 1)
    points_inside = np.sum(distance <= 1)
    
    # Estimate Pi
    pi_estimate = 4 * points_inside / n_points
    
    return pi_estimate, x, y, distance

def visualize_circle_method(n_points=1000, save_path=None):
    """
    Visualize the circle method for estimating Pi.
    
    Args:
        n_points: Number of random points to generate
        save_path: Path to save the figure (optional)
    """
    # Estimate Pi and get points
    pi_estimate, x, y, distance = estimate_pi_circle(n_points)
    
    # Create the plot
    plt.figure(figsize=(10, 10))
    
    # Plot the square
    square = plt.Rectangle((-1, -1), 2, 2, fill=False, color='black', linewidth=2)
    plt.gca().add_patch(square)
    
    # Plot the circle
    circle = plt.Circle((0, 0), 1, fill=False, color='blue', linewidth=2)
    plt.gca().add_patch(circle)
    
    # Plot points
    inside = distance <= 1
    outside = ~inside
    
    plt.scatter(x[inside], y[inside], c='blue', alpha=0.6, label='Inside Circle')
    plt.scatter(x[outside], y[outside], c='red', alpha=0.6, label='Outside Circle')
    
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Circle Method for Estimating π\nPoints: {n_points}, Estimate: {pi_estimate:.6f}')
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    plt.show()
    plt.close()

def circle_convergence_analysis(max_points=100000, step_size=10000, num_trials=3):
    """
    Analyze the convergence of the circle method.
    
    Args:
        max_points: Maximum number of points to use
        step_size: Step size for increasing the number of points
        num_trials: Number of trials for each point count
        
    Returns:
        point_counts: Array of point counts
        estimates: Array of Pi estimates (shape: num_steps x num_trials)
        times: Array of execution times (shape: num_steps x num_trials)
    """
    # Calculate number of steps
    num_steps = max_points // step_size
    
    # Initialize arrays
    point_counts = np.arange(step_size, max_points + step_size, step_size)
    estimates = np.zeros((num_steps, num_trials))
    times = np.zeros((num_steps, num_trials))
    
    # Run trials
    for i, n_points in enumerate(point_counts):
        for j in range(num_trials):
            start_time = time.time()
            pi_estimate, _, _, _ = estimate_pi_circle(n_points)
            end_time = time.time()
            
            estimates[i, j] = pi_estimate
            times[i, j] = end_time - start_time
    
    return point_counts, estimates, times

def plot_circle_convergence(max_points=100000, step_size=10000, num_trials=3, save_path=None):
    """
    Plot the convergence of the circle method.
    
    Args:
        max_points: Maximum number of points to use
        step_size: Step size for increasing the number of points
        num_trials: Number of trials for each point count
        save_path: Path to save the figure (optional)
    """
    # Get convergence data
    point_counts, estimates, _ = circle_convergence_analysis(max_points, step_size, num_trials)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot mean and standard deviation
    mean_estimate = np.mean(estimates, axis=1)
    std_estimate = np.std(estimates, axis=1)
    
    plt.plot(point_counts, mean_estimate, 'b-', linewidth=2, label='Mean Estimate')
    plt.fill_between(point_counts, 
                     mean_estimate - std_estimate, 
                     mean_estimate + std_estimate, 
                     color='b', alpha=0.2, label='±1 Standard Deviation')
    
    # Plot true value of Pi
    plt.axhline(y=np.pi, color='g', linestyle='--', label='True π')
    
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    plt.xlabel('Number of Points')
    plt.ylabel('Estimated π')
    plt.title('Convergence of Circle Method for Estimating π')
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    plt.show()
    plt.close()

# ===== BUFFON'S NEEDLE METHOD =====

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

def visualize_buffon_needle(n_needles, needle_length=1.0, line_spacing=2.0, save_path=None):
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
    
    plt.tight_layout()
    
    if save_path:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    plt.show()
    plt.close()

def buffon_convergence_analysis(max_needles=100000, step_size=10000, num_trials=3):
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

def plot_buffon_convergence(max_needles=100000, step_size=10000, num_trials=3, save_path=None):
    """
    Plot the convergence of Pi estimates as the number of needles increases.
    
    Args:
        max_needles: Maximum number of needles to use
        step_size: Step size for increasing the number of needles
        num_trials: Number of trials for each needle count
        save_path: Path to save the figure (optional)
    """
    # Get convergence data
    needle_counts, pi_estimates, _ = buffon_convergence_analysis(max_needles, step_size, num_trials)
    
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
    
    plt.tight_layout()
    
    if save_path:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    plt.show()
    plt.close()

# ===== COMPARISON OF METHODS =====

def compare_methods(max_points=100000, step_size=10000, num_trials=3, save_path=None):
    """
    Compare the circle method and Buffon's Needle method for estimating Pi.
    
    Args:
        max_points: Maximum number of points/needles to use
        step_size: Step size for increasing the number of points/needles
        num_trials: Number of trials for each point/needle count
        save_path: Path to save the figure (optional)
    """
    # Get convergence data for both methods
    point_counts, circle_estimates, _ = circle_convergence_analysis(max_points, step_size, num_trials)
    needle_counts, buffon_estimates, _ = buffon_convergence_analysis(max_points, step_size, num_trials)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot circle method
    circle_mean = np.mean(circle_estimates, axis=1)
    circle_std = np.std(circle_estimates, axis=1)
    plt.plot(point_counts, circle_mean, 'b-', linewidth=2, label='Circle Method')
    plt.fill_between(point_counts, 
                     circle_mean - circle_std, 
                     circle_mean + circle_std, 
                     color='b', alpha=0.2)
    
    # Plot Buffon's Needle method
    buffon_mean = np.mean(buffon_estimates, axis=1)
    buffon_std = np.std(buffon_estimates, axis=1)
    plt.plot(needle_counts, buffon_mean, 'r-', linewidth=2, label='Buffon\'s Needle')
    plt.fill_between(needle_counts, 
                     buffon_mean - buffon_std, 
                     buffon_mean + buffon_std, 
                     color='r', alpha=0.2)
    
    # Plot true value of Pi
    plt.axhline(y=np.pi, color='g', linestyle='--', label='True π')
    
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    plt.xlabel('Number of Points/Needles')
    plt.ylabel('Estimated π')
    plt.title('Comparison of Monte Carlo Methods for Estimating π')
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    plt.show()
    plt.close()

def compare_efficiency(max_points=100000, step_size=10000, num_trials=3, save_path=None):
    """
    Compare the computational efficiency of the circle method and Buffon's Needle method.
    
    Args:
        max_points: Maximum number of points/needles to use
        step_size: Step size for increasing the number of points/needles
        num_trials: Number of trials for each point/needle count
        save_path: Path to save the figure (optional)
    """
    # Get convergence data for both methods
    point_counts, _, circle_times = circle_convergence_analysis(max_points, step_size, num_trials)
    needle_counts, _, buffon_times = buffon_convergence_analysis(max_points, step_size, num_trials)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot circle method
    circle_mean_time = np.mean(circle_times, axis=1)
    circle_std_time = np.std(circle_times, axis=1)
    plt.plot(point_counts, circle_mean_time, 'b-', linewidth=2, label='Circle Method')
    plt.fill_between(point_counts, 
                     circle_mean_time - circle_std_time, 
                     circle_mean_time + circle_std_time, 
                     color='b', alpha=0.2)
    
    # Plot Buffon's Needle method
    buffon_mean_time = np.mean(buffon_times, axis=1)
    buffon_std_time = np.std(buffon_times, axis=1)
    plt.plot(needle_counts, buffon_mean_time, 'r-', linewidth=2, label='Buffon\'s Needle')
    plt.fill_between(needle_counts, 
                     buffon_mean_time - buffon_std_time, 
                     buffon_mean_time + buffon_std_time, 
                     color='r', alpha=0.2)
    
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.xlabel('Number of Points/Needles')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Computational Efficiency Comparison')
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    plt.show()
    plt.close()

# ===== MAIN FUNCTION =====

def main():
    """
    Main function to run all visualizations and comparisons.
    """
    print("Generating visualizations for Pi estimation methods...")
    
    # Create output directory if it doesn't exist
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Circle method visualization
    print("1. Circle Method Visualization")
    visualize_circle_method(n_points=1000, save_path=os.path.join(output_dir, "circle_monte_carlo.png"))
    
    # Circle method convergence
    print("2. Circle Method Convergence")
    plot_circle_convergence(save_path=os.path.join(output_dir, "circle_convergence.png"))
    
    # Buffon's Needle visualization
    print("3. Buffon's Needle Visualization")
    visualize_buffon_needle(n_needles=100, save_path=os.path.join(output_dir, "buffon_needle.png"))
    
    # Buffon's Needle convergence
    print("4. Buffon's Needle Convergence")
    plot_buffon_convergence(save_path=os.path.join(output_dir, "buffon_convergence.png"))
    
    # Method comparison
    print("5. Method Comparison")
    compare_methods(save_path=os.path.join(output_dir, "method_comparison.png"))
    
    # Efficiency comparison
    print("6. Efficiency Comparison")
    compare_efficiency(save_path=os.path.join(output_dir, "efficiency_comparison.png"))
    
    print("All visualizations completed!")
    print(f"PNG files have been saved to the '{output_dir}' directory.")

if __name__ == "__main__":
    main() 