import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy, pearsonr
from sympy import primerange, divisors
from math import log, e, pi

# --- Configuration ---
N = 1000  # range of integers to analyze
k = 0.313  # golden ratio curvature exponent

# --- Helper Functions ---

def d(n):
    """Return number of positive divisors of n."""
    return len(divisors(n))

def kappa(n):
    """Curvature function κ(n) = d(n) * ln(n + 1) / e²"""
    return d(n) * log(n + 1) / (e ** 2)

def theta_prime(n, k):
    """Modular warp mapping θ′(n) = fractional part of κ(n)^k mod 1"""
    return (kappa(n) ** k) % 1

def zeta_shift(n, k):
    """Zeta-shift transform: Z(n) = n * θ′(n)"""
    return n * theta_prime(n, k)

# --- Generate Data ---

x = np.arange(2, N + 1)
kappa_vals = np.array([kappa(n) for n in x])
theta_vals = np.array([theta_prime(n, k) for n in x])
z_vals = np.array([zeta_shift(n, k) for n in x])

# --- Identify primes for comparison ---
primes = set(primerange(2, N + 1))
is_prime = np.array([int(n in primes) for n in x])

# --- Analysis ---

# Prime vs composite densities in θ′ space
bins = np.linspace(0, 1, 40)
hist_prime, _ = np.histogram(theta_vals[is_prime == 1], bins=bins, density=True)
hist_comp, _ = np.histogram(theta_vals[is_prime == 0], bins=bins, density=True)

# KL Divergence (primes || composites)
kl_divergence = entropy(hist_prime + 1e-12, hist_comp + 1e-12)  # smoothing

# Gini coefficient
def gini(array):
    sorted_arr = np.sort(array)
    n = len(array)
    return (2 * np.sum((np.arange(1, n + 1)) * sorted_arr) / (n * np.sum(sorted_arr))) - (n + 1) / n

gini_prime = gini(hist_prime)
gini_comp = gini(hist_comp)

# --- Plotting ---

plt.figure(figsize=(10, 6))
plt.hist(theta_vals[is_prime == 1], bins=bins, alpha=0.6, label='Primes', density=True)
plt.hist(theta_vals[is_prime == 0], bins=bins, alpha=0.6, label='Composites', density=True)
plt.title(f"θ′(n) Density Distribution (k = {k})")
plt.xlabel("θ′(n)")
plt.ylabel("Density")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Output Summary ---

print(f"KL Divergence (Primes || Composites): {kl_divergence:.4f}")
print(f"Gini Coefficient (Primes): {gini_prime:.4f}")
print(f"Gini Coefficient (Composites): {gini_comp:.4f}")

# Optional: return Z(n) trajectory plot
plt.figure(figsize=(10, 5))
plt.plot(x, z_vals, label="Z(n) = n·θ′(n)", lw=0.8)
plt.title("Zeta-Shift Transform Trajectory Z(n)")
plt.xlabel("n")
plt.ylabel("Z(n)")
plt.grid(True)
plt.tight_layout()
plt.show()

