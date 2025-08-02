import numpy as np
import matplotlib.pyplot as plt
import math
from sympy import sieve as sympy_sieve


# --- Optimized Sieve of Eratosthenes --- #
def compute_primes(limit):
    """
    Computes all primes up to `limit` using an optimized sieve.
    """
    prime_sieve = [True] * (limit + 1)
    prime_sieve[0] = prime_sieve[1] = False  # 0 and 1 are not primes

    for i in range(2, int(math.sqrt(limit)) + 1):
        if prime_sieve[i]:
            for multiple in range(i * i, limit + 1, i):
                prime_sieve[multiple] = False

    # Return list of all primes
    return np.array([i for i, is_prime in enumerate(prime_sieve) if is_prime])


# --- Gap Computations --- #
def compute_gaps_and_predictions(primes):
    """
    Computes actual prime gaps and predicted gaps based on Cramér's conjecture.

    Returns:
        real_gaps: Differences between consecutive primes
        predicted_gaps: Gaps predicted by Cramér's conjecture (log²)
    """
    real_gaps = np.diff(primes)  # Differences between consecutive primes
    predicted_gaps = (np.log(primes[:-1]) ** 2)  # Predicted gaps, removing the last unused prime
    return real_gaps, predicted_gaps


# --- Visualization of Deviations --- #
def plot_prime_news(primes, real_gaps, predicted_gaps):
    """
    Plots deviations between real and predicted gaps as 'prime news' signals.

    Args:
        primes: Array of prime numbers
        real_gaps: Actual gaps between consecutive primes
        predicted_gaps: Predicted gaps based on Cramér's conjecture
    """
    indices = np.arange(1, len(primes))  # Use indices because we're comparing between consecutive primes

    # Compute the deviation
    deviations = np.abs(real_gaps - predicted_gaps)

    # Calculate dynamic threshold based on statistics
    deviation_mean = np.mean(deviations)
    deviation_std = np.std(deviations)
    deviation_threshold = deviation_mean + 1.5 * deviation_std  # Dynamic threshold based on statistical properties

    # Identify significant deviations (prime news)
    significant_indices = deviations > deviation_threshold

    # Create the visualizations
    plt.figure(figsize=(12, 6))

    # Plot real and predicted gaps
    plt.plot(indices, real_gaps, label="Real Gaps", color="blue", lw=1)
    plt.plot(indices, predicted_gaps, label="Predicted Gaps (Cramér)", color="green", lw=1, linestyle="--")

    # Highlight significant deviations
    plt.scatter(
        indices[significant_indices],
        deviations[significant_indices],
        color="red",
        label=f"Prime News (Deviations > {deviation_threshold:.2f})",
        zorder=5,
    )

    # Add annotations for top 3 deviations
    top_indices = np.argsort(deviations)[-3:]
    for idx in top_indices:
        plt.annotate(
            f"p={primes[idx+1]}",
            (indices[idx], deviations[idx]),
            xytext=(10, 10),
            textcoords="offset points",
            arrowprops=dict(arrowstyle='->', color='black')
        )

    # Add labels
    plt.xlabel("Index of Prime")
    plt.ylabel("Gap Size / Deviation")
    plt.title("Real vs Predicted Prime Gaps with Prime News Signals", fontsize=14)
    plt.legend()
    plt.grid(True)

    # Add log scale for x-axis to better visualize the distribution
    plt.xscale('log')

    # Save the figure
    plt.savefig("prime_news_signals.png", dpi=300)
    plt.show()


# --- Alternative Computation Using SymPy --- #
def compute_primes_sympy(limit):
    """
    Computes primes using SymPy's optimized sieve (can be faster for very large limits).
    """
    return np.array(list(sympy_sieve.primerange(2, limit + 1)))


# --- Main Script --- #
if __name__ == "__main__":
    # Set the upper limit for primes
    upper_limit = 10**6  # Adjust depending on computational resources

    # Choose computation method based on limit size
    if upper_limit > 10**7:
        print("Using SymPy's optimized sieve for large limit...")
        primes = compute_primes_sympy(upper_limit)
    else:
        primes = compute_primes(upper_limit)

    print(f"Computed {len(primes)} primes up to {upper_limit}.")

    # Compute actual and predicted prime gaps
    real_gaps, predicted_gaps = compute_gaps_and_predictions(primes)

    # Calculate some statistics
    max_real_gap = np.max(real_gaps)
    max_predicted_gap = np.max(predicted_gaps)
    max_deviation = np.max(np.abs(real_gaps - predicted_gaps))

    print(f"Largest real gap: {max_real_gap}")
    print(f"Largest predicted gap: {max_predicted_gap:.2f}")
    print(f"Maximum deviation: {max_deviation:.2f}")

    # Plot results
    plot_prime_news(primes, real_gaps, predicted_gaps)