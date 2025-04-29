import numpy as np
import matplotlib.pyplot as plt
import time
from buffon_needle import estimate_pi_buffon, convergence_analysis as buffon_convergence
from circle_monte_carlo import estimate_pi_circle, convergence_analysis as circle_convergence

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
    point_counts, circle_estimates, circle_times = circle_convergence(max_points, step_size, num_trials)
    needle_counts, buffon_estimates, buffon_times = buffon_convergence(max_points, step_size, num_trials)
    
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
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.tight_layout()
    plt.show()

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
    point_counts, _, circle_times = circle_convergence(max_points, step_size, num_trials)
    needle_counts, _, buffon_times = buffon_convergence(max_points, step_size, num_trials)
    
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
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Compare methods
    compare_methods(save_path="method_comparison.png")
    
    # Compare efficiency
    compare_efficiency(save_path="efficiency_comparison.png") 