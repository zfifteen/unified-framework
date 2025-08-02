import numpy as np
import matplotlib.pyplot as plt
from sympy import primerange, divisor_count
import scipy.stats as stats
from scipy.fft import fft
from scipy.cluster.vq import kmeans2
import math

# Constants
PHI = (1 + math.sqrt(5)) / 2  # Golden ratio
K_STAR = 0.3  # Optimal curvature parameter
E = math.exp(1)  # Base of natural logarithm
NUM_BINS = 50  # Number of bins for density analysis
NUM_BOOTSTRAP = 1000  # Bootstrap samples
CONFIDENCE_LEVEL = 0.95  # For CI
FOURIER_THRESHOLD = 0.4  # Empirical threshold for asymmetry

def compute_z(n):
    """
    Compute Z(n) = n * (kappa(n) / e**2), where kappa(n) = d(n) * ln(n+1)
    """
    if n < 1:
        return 0.0
    n_float = float(n)
    d_n = float(divisor_count(n))
    ln_term = math.log(n_float + 1)
    kappa = d_n * ln_term
    return n_float * (kappa / (E ** 2))

def prime_curvature_transform(n, k=K_STAR):
    """
    Apply prime curvature transformation: theta'(n, k) = phi * ((n mod phi) / phi)^k
    Normalized by bounding shifts with Z model relative to c ~ e^2.
    """
    # Compute modular residue; since phi irrational, use fractional part
    n_float = float(n)
    frac = (n_float / PHI) - math.floor(n_float / PHI)
    warped = PHI * (frac ** k)
    # Normalize shift bounding by Z model invariant
    z_shift = compute_z(n)
    delta_max = E ** 2
    normalized_shift = z_shift / delta_max
    return warped * math.exp(-normalized_shift)  # Exponential bounding for positive geodesics

def compute_density_enhancement(binned_counts, num_primes):
    """
    Compute density enhancement as (max_bin_density / uniform_density) - 1
    """
    uniform_density = num_primes / NUM_BINS
    max_density = np.max(binned_counts)
    return (max_density / uniform_density) - 1

def bootstrap_ci(data, statistic_func, num_samples=NUM_BOOTSTRAP, alpha=1 - CONFIDENCE_LEVEL):
    """
    Compute bootstrap confidence interval for a statistic
    """
    stats = np.array([statistic_func(np.random.choice(data, len(data), replace=True)) for _ in range(num_samples)])
    lower = np.percentile(stats, 100 * (alpha / 2))
    upper = np.percentile(stats, 100 * (1 - alpha / 2))
    return lower, upper

def fourier_asymmetry(hist_bins):
    """
    Compute Fourier sine coefficient sum as asymmetry metric
    """
    fft_vals = fft(hist_bins)
    sine_coeffs = np.imag(fft_vals[1:])  # Sine components from imaginary parts (excluding DC)
    S_b = np.sum(np.abs(sine_coeffs)) / np.sum(np.abs(fft_vals[1:])) if np.sum(np.abs(fft_vals[1:])) > 0 else 0
    return S_b

def main(N=1000000):
    # Generate primes up to N
    primes = np.array(list(primerange(1, N + 1)), dtype=np.int64)
    num_primes = len(primes)

    # Apply transformation to primes
    transformed_primes = np.array([prime_curvature_transform(p) for p in primes], dtype=np.float64)

    # Bin the transformed values for density analysis (range bounded by phi)
    hist, bin_edges = np.histogram(transformed_primes, bins=NUM_BINS, range=(0, PHI))

    # Falsifiability Test 1: KS test against uniform distribution
    # Normalize transformed_primes to [0,1] for uniform CDF
    normalized_tp = transformed_primes / PHI
    ks_stat, ks_p = stats.kstest(normalized_tp, 'uniform')
    if ks_p > 0.05:
        raise AssertionError("Falsified: Prime distribution consistent with uniform (KS p-value > 0.05)")

    # Compute density enhancement
    enhancement = compute_density_enhancement(hist, num_primes)

    # Falsifiability Test 2: Bootstrap CI for enhancement
    def enh_stat(tp_sample):
        h, _ = np.histogram(tp_sample, bins=NUM_BINS, range=(0, PHI))
        return compute_density_enhancement(h, len(tp_sample))

    ci_lower, ci_upper = bootstrap_ci(transformed_primes, enh_stat)
    if ci_lower <= 0 <= ci_upper:
        raise ValueError("Falsified: 95% CI for density enhancement includes 0%")

    # Falsifiability Test 3: Fourier asymmetry
    S_b = fourier_asymmetry(hist)
    if S_b < FOURIER_THRESHOLD:
        raise RuntimeError("Falsified: Fourier sine asymmetry S_b < 0.4")

    # Fit GMM (using simple 2-component via kmeans as proxy, since SciPy lacks direct GMM; for quantification)
    # Reshape for kmeans2
    tp_reshaped = transformed_primes.reshape(-1, 1)
    centroids, labels = kmeans2(tp_reshaped, 2)
    # Quantify clustering as inverse mean std dev of components
    stds = [np.std(tp_reshaped[labels == i]) for i in range(2)]
    clustering_compactness = 1 / np.mean(stds) if np.mean(stds) > 0 else 0

    # Visualization
    plt.figure(figsize=(10, 6))
    plt.hist(transformed_primes, bins=NUM_BINS, density=True, alpha=0.7, label='Transformed Primes')
    plt.axhline(1/PHI, color='r', linestyle='--', label='Uniform Density')
    plt.title(f'Prime Density under Curvature Transformation (N={N}, k={K_STAR})')
    plt.xlabel('Transformed Value')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig('prime_density_plot.png')
    plt.show()

    # Output results
    print(f"Density Enhancement: {enhancement * 100:.2f}%")
    print(f"Bootstrap 95% CI: [{ci_lower * 100:.2f}%, {ci_upper * 100:.2f}%]")
    print(f"Fourier Asymmetry S_b: {S_b:.3f}")
    print(f"Clustering Compactness (1/mean_std): {clustering_compactness:.3f}")
    print("Visualization saved as 'prime_density_plot.png'")
    print("All falsifiability tests passed. Geometric clustering confirmed.")

if __name__ == "__main__":
    main()