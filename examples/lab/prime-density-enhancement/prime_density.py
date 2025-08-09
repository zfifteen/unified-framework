import numpy as np
import matplotlib.pyplot as plt
from sympy import primerange
import scipy.stats as stats
import math
import sys
from sklearn.mixture import GaussianMixture
from scipy.fft import fft  # retained in case you want to reintroduce Fourier later

# Seed everything for reproducibility
np.random.seed(42)

# Constants
PHI = (1 + math.sqrt(5)) / 2
K_STAR = 3.33
E = math.exp(1)
NUM_BINS = 100
NUM_BOOTSTRAP = 1000
CONFIDENCE_LEVEL = 0.95
KL_THRESHOLD = 0.1  # Minimum KL divergence to reject uniformity
EPS = 1e-9

def precompute_divisor_counts(N):
    dcount = np.zeros(N + 1, dtype=int)
    for i in range(1, N + 1):
        dcount[i::i] += 1
    return dcount

def compute_z(n, dcount):
    if n <= 1:
        return 0.0
    n_float = float(n)
    d_n = float(dcount[n])
    ln_term = math.log(n_float + 1)
    kappa = d_n * ln_term
    return n_float * (kappa / (E ** 2))

def prime_curvature_transform(n, dcount, k=K_STAR):
    frac = math.modf(n / PHI)[0]
    warped = PHI * (frac ** k)
    z_shift = compute_z(n, dcount)
    return warped * math.exp(-z_shift)

def compute_density_enhancement(binned_counts, num_primes):
    uniform_density = num_primes / NUM_BINS
    max_density = np.max(binned_counts)
    return (max_density / uniform_density) - 1

def bootstrap_ci(data, statistic_func, num_samples=NUM_BOOTSTRAP, alpha=1 - CONFIDENCE_LEVEL):
    n = len(data)
    stats_arr = []
    for _ in range(num_samples):
        idxs = np.random.choice(n, n, replace=True)
        sample = data[idxs]
        stats_arr.append(statistic_func(sample))
    lower = np.percentile(stats_arr, 100 * (alpha / 2))
    upper = np.percentile(stats_arr, 100 * (1 - alpha / 2))
    return lower, upper

def kl_divergence(p, q):
    p = p / np.sum(p)
    q = q / np.sum(q)
    mask = (p > 0) & (q > 0)
    return np.sum(p[mask] * np.log2(p[mask] / q[mask]))

def main(N=1_000_000):
    dcount = precompute_divisor_counts(N)
    primes = np.array(list(primerange(1, N + 1)), dtype=int)
    num_primes = len(primes)
    transformed = np.array([prime_curvature_transform(p, dcount) for p in primes])

    hist, edges = np.histogram(transformed,
                               bins=NUM_BINS,
                               range=(-EPS, PHI + EPS))

    test_results = []

    # KS Test
    try:
        normalized = transformed / PHI
        ks_stat, ks_p = stats.kstest(normalized, 'uniform', args=(0, 1))
        if ks_p > 0.05:
            raise AssertionError(f"KS p-value {ks_p:.4f} > 0.05")
        test_results.append(("KS test", True, f"p-value={ks_p:.4f}"))
    except AssertionError as e:
        test_results.append(("KS test", False, str(e)))

    enhancement = compute_density_enhancement(hist, num_primes)

    # Bootstrap CI
    def enh_stat(sample):
        h, _ = np.histogram(sample, bins=NUM_BINS, range=(-EPS, PHI + EPS))
        return compute_density_enhancement(h, len(sample))

    try:
        ci_low, ci_high = bootstrap_ci(transformed, enh_stat)
        if ci_low <= 0 <= ci_high:
            raise ValueError(f"CI [{ci_low:.4f}, {ci_high:.4f}] includes 0")
        test_results.append(("Bootstrap CI", True, f"[{ci_low:.4f}, {ci_high:.4f}]"))
    except ValueError as e:
        test_results.append(("Bootstrap CI", False, str(e)))

    # KL Divergence Test
    try:
        p = hist
        q = np.ones_like(p)  # uniform reference
        kl = kl_divergence(p, q)
        if kl < KL_THRESHOLD:
            raise RuntimeError(f"KL divergence {kl:.4f} < threshold {KL_THRESHOLD}")
        test_results.append(("KL divergence", True, f"KL={kl:.4f}"))
    except RuntimeError as e:
        test_results.append(("KL divergence", False, str(e)))

    # Clustering via GaussianMixture
    tp_reshaped = transformed.reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, random_state=42)
    gmm.fit(tp_reshaped)
    labels = gmm.predict(tp_reshaped)
    stds = [np.std(tp_reshaped[labels == i]) for i in range(2)]
    clustering_compactness = 1 / np.mean(stds) if np.mean(stds) > 0 else float('inf')

    # Visualization
    plt.figure(figsize=(10, 6))
    plt.hist(transformed, bins=NUM_BINS, density=True, alpha=0.7,
             label='Transformed Primes')
    plt.axhline(1 / PHI, color='r', linestyle='--', label='Uniform Density')
    plt.title(f'Prime Density under Curvature Transform (N={N}, k={K_STAR})')
    plt.xlabel('Transformed Value')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig('prime_density_plot.png')
    plt.close()

    # Report
    print("\n=== Falsifiability Test Results ===")
    for name, passed, info in test_results:
        status = "PASS" if passed else "FAIL"
        print(f"{name:20s}: {status} ({info})")

    print("\n=== Metrics ===")
    print(f"Density Enhancement      : {enhancement*100:.2f}%")
    print(f"Clustering Compactness   : {clustering_compactness:.3f}")
    print("Visualization saved as 'prime_density_plot.png'")

    if not all(passed for _, passed, _ in test_results):
        sys.exit(1)

if __name__ == "__main__":
    main()
