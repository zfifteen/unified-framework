import numpy as np
import matplotlib.pyplot as plt
import math
from sympy import sieve


# --- Optimized Sieve of Eratosthenes --- #
def compute_primes(limit):
    """
    Computes all primes up to `limit` using an optimized sieve.
    """
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False  # 0 and 1 are not primes
    
    for i in range(2, int(math.sqrt(limit)) + 1):
        if sieve[i]:
            for multiple in range(i * i, limit + 1, i):
                sieve[multiple] = False

    # Return list of all primes
    return np.array([i for i, is_prime in enumerate(sieve) if is_prime])


# --- Gap Computations --- #
def compute_gaps_and_predictions(primes):
    """
    Computes actual prime gaps and predicted gaps based on Cramér's conjecture.
    """
    real_gaps = np.diff(primes)  # Differences between consecutive primes
    predicted_gaps = (np.log(primes[:-1]) ** 2)  # Predicted gaps, removing the last unused prime
    return real_gaps, predicted_gaps


def plot_prime_news(primes, real_gaps, predicted_gaps, save_path="prime_news_plot.png"):
    """
    Plots deviations between real and predicted gaps as 'prime news' signals and saves the plot to a file.

    Args:
        primes (np.array): Array of prime numbers.
        real_gaps (np.array): Array of actual gaps between consecutive primes.
        predicted_gaps (np.array): Array of predicted prime gaps based on Cramér's conjecture.
        save_path (str): File path to save the resulting plot as an image.
    """
    indices = np.arange(1, len(primes))  # Use indices because we're comparing between consecutive primes
    
    # Compute the deviation
    deviations = np.abs(real_gaps - predicted_gaps)
    deviation_threshold = 1.5  # Threshold for highlighting significant outliers

    # Identify significant deviations (prime news)
    significant_indices = deviations > deviation_threshold

    # Create the visualization
    plt.figure(figsize=(12, 6))
    
    # Plot real and predicted gaps
    plt.plot(indices, real_gaps, label="Real Gaps", color="blue", lw=1, alpha=0.8)
    plt.plot(indices, predicted_gaps, label="Predicted Gaps (Cramér)", color="green", lw=1, linestyle="--", alpha=0.8)

    # Highlight significant deviations
    plt.scatter(
        indices[significant_indices],
        deviations[significant_indices],
        color="red",
        label="Prime News (Large Deviations)",
        zorder=5,
    )

    # Customize axes to emphasize meaningful values
    plt.xlim(0, min(1000, len(indices)))  # Show meaningful data on the left
    plt.ylim(0, max(max(real_gaps), max(predicted_gaps)) * 1.1)

    # Add labels, legend, and title
    plt.xlabel("Prime Index", fontsize=12)
    plt.ylabel("Gap Size / Deviation", fontsize=12)
    plt.title("Real vs Predicted Prime Gaps with Prime News Signals", fontsize=14)
    plt.legend(loc="upper left")
    plt.grid(True)

    # Save the plot to a file
    plt.savefig(save_path)
    print(f"Plot successfully saved to {save_path}")

    plt.close()


# --- Main Script --- #
if __name__ == "__main__":
    # Set the upper limit for primes
    upper_limit = 10**6  # Adjust depending on computational resources

    # Precompute primes
    primes = compute_primes(upper_limit)
    print(f"Computed {len(primes)} primes up to {upper_limit}.")

    # Compute actual and predicted prime gaps
    real_gaps, predicted_gaps = compute_gaps_and_predictions(primes)
    
    # Plot results
    plot_prime_news(primes, real_gaps, predicted_gaps)