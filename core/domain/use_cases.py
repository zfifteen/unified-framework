import domain
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import argparse

"""
for the first 50 numbers
- instantiate a PhysicalZetaShift z(n) for that number
- print out the number n
- print out the z(n) value
- export results to a CSV file

"""
# todo: Enhance the 3D plot by adding a colorbar for the Z(n) cmap, highlighting prime points in red (using a primality mask on scatter), and enabling gridlines or rotation views, to better visualize curvature anomalies in the numerical landscape per the universal frame shift transformer.
# Importing the required class
from domain import PhysicalZetaShift

# --- Helper function to generate all primes up to a limit ---
def sieve_of_eratosthenes(limit):
    """Returns a list of all primes up to a given limit using the Sieve of Eratosthenes."""
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False  # 0 and 1 are not prime numbers
    
    for num in range(2, int(limit ** 0.5) + 1):
        if sieve[num]:
            for multiple in range(num * num, limit + 1, num):
                sieve[multiple] = False
                
    return [n for n, is_prime in enumerate(sieve) if is_prime]

# --- Argument Parsing for Configurable Upper Limit ---
def parse_arguments():
    """Parses command-line arguments for configurable parameters."""
    parser = argparse.ArgumentParser(description="Process Zeta Shift parameters.")
    parser.add_argument(
        "--limit", type=int, default=50, help="Upper limit for processing numbers (default: 50)"
    )
    return parser.parse_args()

# --- Visualization Function ---
def plot_3d_results(results, primes):
    """
    Enhanced 3D plot with the following features:
    - Colorbar for Z(n)
    - Prime points highlighted in red
    - Gridlines and rotation enabled for better visualization
    """
    # Extract data for plotting
    numbers = np.array([row[0] for row in results])  # n
    z_values = np.array([row[1] for row in results])  # Z(n)
    gaps = np.array([row[2] for row in results])  # Largest gap (v)

    # Create figure and 3D axis
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Establish a colormap for Z(n), mapping its values to colors
    cmap = plt.get_cmap("viridis")
    colors = cmap((z_values - z_values.min()) / (z_values.max() - z_values.min()))

    # Highlight numbers that are primes
    is_prime = np.array([n in primes for n in numbers])  # Boolean mask
    non_prime_mask = ~is_prime

    # Scatter plot for non-prime points (colored by Z(n))
    scatter = ax.scatter(
        numbers[non_prime_mask],  # X-axis: numbers
        gaps[non_prime_mask],     # Y-axis: gaps
        z_values[non_prime_mask], # Z-axis: Z(n)
        c=colors[non_prime_mask], # Color from colormap
        marker='o',
        label="Non-prime"
    )

    # Highlight prime points in red
    ax.scatter(
        numbers[is_prime],
        gaps[is_prime],
        z_values[is_prime],
        c="red",  # Constant red color for primes
        marker="^",
        label="Primes"
    )

    # Add labels and title
    ax.set_xlabel("Number (n)")
    ax.set_ylabel("Largest Gap (v)")
    ax.set_zlabel("Z(n)")
    ax.set_title("3D Visualization of Z(n) with Prime Highlighting", fontsize=14)

    # Enable gridlines
    ax.grid(True)

    # Add colorbar for Z(n)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=z_values.min(), vmax=z_values.max()))
    sm.set_array([])  # Required for colorbar
    cbar = plt.colorbar(sm, ax=ax, shrink=0.5, aspect=20)
    cbar.set_label("Z(n) Value", fontsize=12)

    # Enable rotation
    ax.view_init(elev=30, azim=45)  # Elevation and Azimuthal angle

    # Add a legend
    ax.legend()

    # Show the plot
    plt.show()

# --- Main Execution ---
if __name__ == "__main__":
    args = parse_arguments()
    upper_limit = args.limit

    # Ensure the limit is reasonable
    if upper_limit <= 0:
        raise ValueError("The upper limit must be a positive integer.")

    # Precompute primes up to the specified upper limit using the Sieve of Eratosthenes
    primes = set(sieve_of_eratosthenes(upper_limit))

    # Variables to track prime gaps
    largest_gap = 0  # This will now only increment
    last_prime = None

    # Container for results
    results = []

    # Loop for the specified range of numbers
    for n in range(1, upper_limit + 1):
        if n in primes:
            if last_prime is not None:
                gap = n - last_prime
                if gap > largest_gap:
                    largest_gap = gap  # Update only if a new larger gap is found
            last_prime = n
        # Note: No need to assign gap here again since it's done above

        # Instantiate PhysicalZetaShift with the corrected largest_gap
        zeta_shift = PhysicalZetaShift(t=n, v=largest_gap)
        z_value = zeta_shift.compute_z()

        # Collect results
        results.append([n, z_value, largest_gap])

        # Print results to the console
        print(f"Number n: {n}, Z(n): {z_value}, Largest Gap (v): {largest_gap}")

    # Export results to CSV
    csv_filename = f"zeta_shift_results_up_to_{upper_limit}.csv"
    with open(csv_filename, "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Number (n)", "Z(n) Value", "Largest Gap (v)"])  # Header row
        csv_writer.writerows(results)

    print(f"\nResults successfully exported to '{csv_filename}'.")

    # Assuming results and primes are already computed as previously:
    # results: List of rows [n, Z(n), largest_gap]
    # primes: Set of all primes up to the limit
    plot_3d_results(results, primes)
import numpy as np
import matplotlib.pyplot as plt
from domain import PhysicalZetaShift
from scipy.special import loggamma

def hyperbolic_projection(Z):
    """
    Apply hyperbolic projection to curve Z values using a normalized tanh function.
    """
    Z_max = np.max(Z)
    return np.tanh(Z / Z_max) if Z_max > 0 else Z

def compute_dynamic_delta_max(n_values):
    """
    Compute Δmax dynamically based on the prime number theorem approximation: log(n).
    """
    return np.log(n_values)

def create_line_plot_with_projection(n_values, Z_values, primes):
    """
    Replace scatter plot with a connected line plot where Z is adjusted using hyperbolic projection.
    Prime numbers are marked separately for enhanced visualization.
    """
    # Compute hyperbolic projections for Z
    Z_projected = hyperbolic_projection(Z_values)

    # Create mask for prime numbers
    is_prime = np.array([n in primes for n in n_values])

    # Plot the connected line plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(n_values, Z_projected, label="Curve through Z values", color="blue", linewidth=2)

    # Highlight prime numbers on the line plot
    ax.scatter(n_values[is_prime], Z_projected[is_prime], color="red", label="Primes", zorder=5)

    # Add labels and legend
    ax.set_xlabel("n (Numbers)", fontsize=12)
    ax.set_ylabel("Projected Z(n)", fontsize=12)
    ax.set_title("Connected Line Plot with Hyperbolic Projection for Z(n)", fontsize=14)
    ax.legend()
    ax.grid(True)
    plt.show()

if __name__ == "__main__":
    # Generate numbers and compute their physical Zeta shifts
    upper_limit = 50
    primes = set(sieve_of_eratosthenes(upper_limit))
    n_values = np.arange(1, upper_limit + 1)

    # Compute dynamic Δmax values based on prime number theorem approximation
    delta_max_values = compute_dynamic_delta_max(n_values)

    # Compute Z(n) values using PhysicalZetaShift
    Z_values = []
    for n in n_values:
        delta_max = delta_max_values[n - 1]
        zeta_shift = PhysicalZetaShift(t=n, v=delta_max)
        Z_values.append(zeta_shift.compute_z())

    # Replace scatter plot with the desired line plot
    create_line_plot_with_projection(n_values, np.array(Z_values), primes)