# Import necessary libraries
import numpy as np  # For numerical operations, especially with arrays and mathematical constants
import matplotlib.pyplot as plt  # For plotting data, such as histograms
import itertools  # For creating combinations of elements, used here for planet pairs
import math  # For basic mathematical functions (not strictly used here as np.pi is used)

# --- Constants ---
# Define fundamental mathematical and physical constants used in calculations.

# The Golden Ratio, an irrational number approximately 1.618, often found in nature.
PHI = (1 + np.sqrt(5)) / 2
# The mathematical constant Pi, the ratio of a circle's circumference to its diameter.
PI = np.pi
# The base of the natural logarithm 'e' squared. Used in the curvature calculation.
E_SQUARED = np.exp(2)

# --- Data ---
# A dictionary containing the orbital periods of the planets in our solar system, measured in Earth years.
# This data is the basis for the analysis.
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

# --- Functions for Transformations ---
# These functions define mathematical transformations that are applied to the data.

# Function to compute the curvature κ(n) for a given integer n.
# The curvature is a function of the number of divisors of n and the natural log of (n+1).
def curvature(n):
    # Calculate d_n, the number of divisors of n.
    d_n = len([i for i in range(1, n + 1) if n % i == 0])
    # Return the curvature value, normalized by e^2.
    return d_n * np.log(n + 1) / E_SQUARED

# Z transformation, which scales a number 'n' based on its curvature 'kappa'.
# This can be seen as a normalization or transformation of 'n'.
def Z(n, kappa=None):
    # If kappa is not provided, calculate it using the curvature function.
    if kappa is None:
        kappa = curvature(n)
    # Apply the Z transformation formula.
    return n / np.exp(kappa)

# θ'(n, k) transformation, a function involving the Golden Ratio (phi).
# This function calculates a value based on the remainder of n divided by phi.
def theta_prime(n, k, phi=PHI):
    # The formula involves modular arithmetic with phi.
    return phi * ((n % phi) / phi) ** k

# --- Analysis Functions ---
# These functions perform the core analysis of the orbital period data.

# Generates pairwise ratios of the orbital periods for all combinations of planets.
def pairwise_ratios(periods):
    # Create all unique pairs of planets from the periods dictionary.
    pairs = list(itertools.combinations(periods.items(), 2))
    ratios = {}
    # Iterate through each pair to calculate the ratio of their orbital periods.
    for (a_name, a), (b_name, b) in pairs:
        # Calculate the ratio, ensuring it's always >= 1 for consistency.
        ratio = max(a / b, b / a)
        # Store the ratio with the names of the two planets as the key.
        ratios[(a_name, b_name)] = ratio
    return ratios

# Compares the calculated ratios to a set of target mathematical constants.
def compare_to_constants(ratios, constants, tolerance=0.05):
    results = []
    # Iterate over all calculated ratios.
    for pair, ratio in ratios.items():
        # For each ratio, iterate over all target constants to check for a match.
        for label, const in constants.items():
            # Calculate the relative error between the ratio and the constant.
            rel_error = abs(ratio - const) / const
            # If the error is within the specified tolerance, it's considered a match.
            if rel_error < tolerance:
                # Add the match details to the results list.
                results.append((pair, ratio, label, const, rel_error))
    return results

# --- Main Execution ---
# The main part of the script that orchestrates the analysis and output.

# A dictionary of target constants to which the orbital ratios will be compared.
# These include well-known mathematical constants and simple integer ratios.
target_constants = {
    "phi": PHI,
    "pi/phi": PI / PHI,
    "2": 2.0,
    "3/2": 1.5,
    "5/3": 5/3,
    "8/5": 8/5
}

# Perform the main analysis by first calculating the ratios and then finding matches.
ratios = pairwise_ratios(orbital_periods)
matches = compare_to_constants(ratios, target_constants)

# --- Output ---
# Display the results of the analysis to the user.

print("--- Orbital Ratio Analysis Results ---")
# Iterate through the list of matches and print them in a readable format.
for match in matches:
    print(f"Pair: {match[0]}, Ratio: {match[1]:.3f}, Matches: {match[2]} ≈ {match[3]:.3f} (Error: {match[4]*100:.2f}%)")
print("------------------------------------")


# --- Optional Visualization ---
# This section plots a histogram of the orbital period ratios for visual analysis.

# Create a new figure for the plot with a specified size.
plt.figure(figsize=(10, 6))
# Get all the calculated ratios as a list.
all_ratios = list(ratios.values())
# Create a histogram of the ratios to show their distribution.
plt.hist(all_ratios, bins=np.linspace(1, max(all_ratios), 30), alpha=0.7, label="Orbital Ratios")
# Add vertical lines for each of the target constants to see how they align with the ratio distribution.
for name, val in target_constants.items():
    plt.axvline(val, color='r', linestyle='--', label=f"{name} ≈ {val:.3f}")
# Set the title and labels for the plot.
plt.title("Distribution of Planetary Orbital Period Ratios")
plt.xlabel("Orbital Period Ratio")
plt.ylabel("Frequency")
# Add a legend to identify the different elements of the plot.
plt.legend()
# Add a grid for better readability.
plt.grid(True)
# Adjust the plot to ensure everything fits without overlapping.
plt.tight_layout()
# Display the plot.
plt.show()
