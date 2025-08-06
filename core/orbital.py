# Let's begin building a refactored version of the original script with added visualizations and analysis.
# This version will include:
# - Pairwise period ratio analysis
# - Z-transformation mapping
# - θ′ transformation plots
# - Distribution of curvature values
# - Heatmap and scatterplot outputs

from math import isclose
import numpy as np
import matplotlib.pyplot as plt
import itertools

# Constants
PHI = (1 + np.sqrt(5)) / 2
E_SQUARED = np.exp(2)

# Orbital periods in Earth years
orbital_periods = {
    "Mercury": 0.241,
    "Venus": 0.615,
    "Earth": 1.000,
    "Mars": 1.881,
    "Jupiter": 11.863,
    "Saturn": 29.447,
    "Uranus": 84.017,
    "Neptune": 164.791
}

# Functions
def curvature(n):
    n_int = int(round(n))
    d_n = len([i for i in range(1, n_int + 1) if n_int % i == 0])
    return d_n * np.log(n_int + 1) / E_SQUARED

def Z(n, kappa=None):
    if kappa is None:
        kappa = curvature(n)
    return n / np.exp(kappa)

def theta_prime(n, k, phi=PHI):
    return phi * ((n % phi) / phi) ** k

def pairwise_ratios(periods):
    pairs = list(itertools.combinations(periods.items(), 2))
    data = []
    for (a_name, a), (b_name, b) in pairs:
        ratio = max(a / b, b / a)
        label = f"{a_name}-{b_name}"
        data.append((label, ratio))
    return data

# Compute values
ratios_data = pairwise_ratios(orbital_periods)
labels, ratios = zip(*ratios_data)
curvatures = [curvature(r) for r in ratios]
z_values = [Z(r, kappa=k) for r, k in zip(ratios, curvatures)]
theta_values = [theta_prime(r, k=0.3) for r in ratios]

# Plot 1: Bar plot of period ratios
plt.figure(figsize=(10, 6))
plt.bar(labels, ratios, color='skyblue')
plt.xticks(rotation=90)
plt.title("Pairwise Orbital Period Ratios")
plt.ylabel("Ratio (max(a/b, b/a))")
plt.tight_layout()
plt.show()

# Plot 2: Z-transformed ratios
plt.figure(figsize=(10, 6))
plt.bar(labels, z_values, color='orange')
plt.xticks(rotation=90)
plt.title("Z-Transformed Orbital Period Ratios")
plt.ylabel("Z(ratio)")
plt.tight_layout()
plt.show()

# Plot 3: θ′ values
plt.figure(figsize=(10, 6))
plt.bar(labels, theta_values, color='green')
plt.xticks(rotation=90)
plt.title(r"θ′(n, k≈0.3) using Golden Ratio Modulo")
plt.ylabel(r"θ′(ratio)")
plt.tight_layout()
plt.show()

# Plot 4: Scatter Z vs Curvature
plt.figure(figsize=(8, 6))
plt.scatter(curvatures, z_values, c='purple')
plt.title("Z(ratio) vs Curvature")
plt.xlabel("Curvature")
plt.ylabel("Z(ratio)")
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot 5: Histogram of curvature values
plt.figure(figsize=(8, 5))
plt.hist(curvatures, bins=10, color='red', edgecolor='black')
plt.title("Distribution of Curvature Values")
plt.xlabel("Curvature")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()
