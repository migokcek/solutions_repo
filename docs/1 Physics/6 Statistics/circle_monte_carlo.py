import numpy as np
import matplotlib.pyplot as plt
import time

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
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.tight_layout()
    plt.show()

def convergence_analysis(max_points=100000, step_size=10000, num_trials=3):
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

def plot_convergence(max_points=100000, step_size=10000, num_trials=3, save_path=None):
    """
    Plot the convergence of the circle method.
    
    Args:
        max_points: Maximum number of points to use
        step_size: Step size for increasing the number of points
        num_trials: Number of trials for each point count
        save_path: Path to save the figure (optional)
    """
    # Get convergence data
    point_counts, estimates, _ = convergence_analysis(max_points, step_size, num_trials)
    
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
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Visualize the circle method
    visualize_circle_method(save_path="circle_monte_carlo.png")
    
    # Plot convergence
    plot_convergence(save_path="circle_convergence.png") 