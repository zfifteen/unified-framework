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
# todo: fix largest gap, v so that it only increments - the purpose is to track the largest gap encountered on the number line. It should never reset.
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