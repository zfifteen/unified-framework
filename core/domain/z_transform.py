"""
Prime Curvature Z-Transform Analysis
===================================

This module implements the Universal Form Z = A(B/C) to analyze prime distributions
through frame-shift transformations, connecting physical and discrete domains.

Key features:
1. Prime Gap Deviation (velocity profile)
2. Zeta Shift (frame curvature)
3. Geometric Projection (3D topology)
4. Invariant Frame Map (meta-analysis)

Usage: python z_transform.py
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sympy import isprime, sieve
from sklearn.mixture import GaussianMixture
import warnings
from matplotlib.cm import plasma
import math
from scipy.spatial import distance

warnings.filterwarnings("ignore")

# Constants
PHI = (1 + np.sqrt(5)) / 2  # Golden ratio
E = np.e  # Euler's number
N_MAX = 1000  # Maximum number to analyze
HELIX_FREQ = 0.1003033  # Optimal frequency from hologram.py
K_STAR = 0.3  # Optimal curvature exponent from proof.py

class ZTransform:
    """
    Implements the universal Z-transform across domains.
    Z = A(B/C) where:
    - A is the observed quantity (e.g., n for integers)
    - B is the rate (e.g., gap to next prime)
    - C is the invariant limit (e.g., max theoretical gap)
    """

    def __init__(self, max_n=N_MAX):
        """Initialize with maximum number to analyze."""
        self.max_n = max_n
        self.numbers = np.arange(1, max_n + 1)
        self.primes = self._compute_primes()
        self.phi = PHI
        self.e = E

    def _compute_primes(self):
        """Compute primes up to max_n using sympy's sieve."""
        return np.array(list(sieve.primerange(2, self.max_n + 1)))

    def compute_prime_gaps(self):
        """Compute gaps between consecutive primes."""
        return np.diff(self.primes)

    def compute_predicted_gaps(self):
        """Compute predicted gaps based on log law."""
        return np.log(self.primes[:-1])

    def compute_gap_deviations(self):
        """Compute deviations between actual and predicted gaps."""
        real_gaps = self.compute_prime_gaps()
        predicted_gaps = self.compute_predicted_gaps()
        return real_gaps - predicted_gaps

    def compute_z_values(self):
        """
        Compute Z-transform values for all numbers.
        Z(n) = n * (Δₙ/Δmax) where Δₙ is gap to next prime or gap value.
        """
        z_values = np.zeros_like(self.numbers, dtype=float)

        # Precompute next prime for each number
        next_prime = np.zeros_like(self.numbers, dtype=int)
        prime_idx = 0

        for i, n in enumerate(self.numbers):
            # Find next prime
            while prime_idx < len(self.primes) and self.primes[prime_idx] <= n:
                prime_idx += 1

            if prime_idx < len(self.primes):
                next_prime_val = self.primes[prime_idx]
                gap = next_prime_val - n
            else:
                # For numbers beyond the largest computed prime
                gap = int(np.log(n)**2)  # Estimate based on Cramér's conjecture

            # Normalize by max theoretical gap (using log squared as upper bound)
            max_gap = np.log(n)**2

            # Compute Z value
            z_values[i-1] = n * (gap / max_gap)

        return z_values

    def frame_shift_residues(self, n_vals, k=K_STAR):
        """
        Apply golden ratio-based curvature transformation.
        θ' = φ * ((n mod φ) / φ) ** k
        """
        mod_phi = np.mod(n_vals, self.phi) / self.phi
        return self.phi * np.power(mod_phi, k)

    def compute_density_map(self, coordinates):
        """
        Compute density map based on inverse distance between points.
        """
        # Use subset for efficiency in distance calculations
        sample_size = min(500, len(coordinates))
        if len(coordinates) > sample_size:
            indices = np.random.choice(len(coordinates), sample_size, replace=False)
            sample_coords = coordinates[indices]
        else:
            sample_coords = coordinates

        # Compute pairwise distances
        dist_matrix = distance.cdist(coordinates, sample_coords)

        # Inverse distance as density (avoid division by zero)
        inv_dist = 1.0 / (dist_matrix + 1e-10)

        # Sum inverse distances for each point
        density = np.sum(inv_dist, axis=1)

        # Normalize
        if np.max(density) > 0:
            density = density / np.max(density)

        return density

    def project_to_3d(self, numbers, helix_freq=HELIX_FREQ):
        """
        Project numbers to 3D space using transformations.
        """
        # Apply curvature transformation
        theta = self.frame_shift_residues(numbers)

        # Create spiral coordinates
        angle = numbers * helix_freq * np.pi
        radius = np.log(numbers + 1)  # log scale for better visualization

        # Compute 3D coordinates
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        z = theta  # Use transformed values for z-coordinate

        return np.column_stack([x, y, z])

    def lorentz_transform(self, velocity, z_coord, c_analog=E):
        """
        Apply Lorentz-like transform to z-coordinates.
        z' = z / sqrt(1 - (v/c)²)
        """
        # Ensure velocity doesn't exceed c_analog
        safe_velocity = np.minimum(velocity, 0.99 * c_analog)

        # Apply transform
        gamma = 1.0 / np.sqrt(1.0 - (safe_velocity/c_analog)**2)
        return z_coord * gamma

    def plot_gap_deviation(self):
        """
        Plot 1: Prime Gap Deviation (velocity profile)
        """
        # Compute gap metrics
        real_gaps = self.compute_prime_gaps()
        predicted_gaps = self.compute_predicted_gaps()
        deviations = np.abs(real_gaps - predicted_gaps)

        # Compute "velocity" as gap divided by log
        velocity = real_gaps / np.log(self.primes[:-1])

        # Identify significant deviations
        deviation_threshold = 1.5
        significant = deviations > deviation_threshold

        # Create plot
        plt.figure(figsize=(12, 6))

        # Plot real and predicted gaps
        plt.plot(self.primes[:-1], real_gaps, label="Real Gaps", color="blue", lw=1)
        plt.plot(self.primes[:-1], predicted_gaps, label="Predicted Gaps",
                 color="green", lw=1, linestyle="--")

        # Highlight significant deviations
        plt.scatter(
            self.primes[:-1][significant],
            deviations[significant],
            color="red",
            label=f"Significant Deviations (>{deviation_threshold})",
            zorder=5,
        )

        # Apply geometric scaling to y-axis
        plt.yscale('log')

        # Add labels and title
        plt.xlabel("Prime Number")
        plt.ylabel("Gap Size / Deviation (log scale)")
        plt.title("Prime Gap Deviation (Velocity Profile)", fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Save plot
        plt.savefig("gap_deviation.png", dpi=300, bbox_inches="tight")
        plt.close()

        # Return metrics for meta-analysis
        return {
            'n': self.primes[1:],  # Skip 2 since we use gaps starting from 3
            'deviation_score': deviations,
            'gap_velocity': velocity
        }

    def plot_zeta_shift(self):
        """
        Plot 2: Zeta Shift (Frame Curvature Map)
        """
        # Compute Z values
        z_values = self.compute_z_values()

        # Compute shift ratios for primes
        prime_indices = np.array([i-1 for i in self.primes if i <= self.max_n])
        prime_z = z_values[prime_indices]

        # Get gaps between consecutive primes
        gaps = self.compute_prime_gaps()
        max_gap = np.max(gaps)
        shift_ratio = gaps / max_gap

        # Create plot
        plt.figure(figsize=(12, 6))

        # Plot Z values for all numbers
        plt.plot(self.numbers, z_values, label="Z(n) Values", color="blue", alpha=0.5)

        # Highlight Z values for primes
        plt.scatter(
            self.primes,
            prime_z,
            color="red",
            marker="*",
            s=100,
            label="Z(p) for Primes",
            zorder=5,
        )

        # Apply hyperbolic transform to emphasize curvature
        #plt.yscale('symlog')  # Symmetric log scale

        # Add labels and title
        plt.xlabel("Number (n)")
        plt.ylabel("Z(n) = n * (Δₙ/Δmax)")
        plt.title("Zeta Shift (Frame Curvature Map)", fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Save plot
        plt.savefig("zeta_shift.png", dpi=300, bbox_inches="tight")
        plt.close()

        # Return metrics for meta-analysis
        return {
            'n': self.numbers,
            'z_value': z_values,
            'shift_ratio': np.interp(self.numbers, self.primes[:-1], shift_ratio)
        }

    def plot_geometric_projection(self):
        """
        Plot 3: Geometric Projection (3D Topology)
        """
        # Project numbers to 3D space
        coords = self.project_to_3d(self.numbers)

        # Identify prime coordinates
        prime_indices = np.array([n-1 for n in self.primes if n <= self.max_n])
        prime_coords = coords[prime_indices]

        # Compute density map
        density = self.compute_density_map(coords)

        # Create 3D plot
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Plot all numbers
        scatter = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            coords[:, 2],
            c=density,
            cmap='viridis',
            alpha=0.7,
            s=10,
        )

        # Highlight primes
        ax.scatter(
            prime_coords[:, 0],
            prime_coords[:, 1],
            prime_coords[:, 2],
            color="red",
            marker="*",
            s=100,
            label="Primes",
        )

        # Add color bar
        plt.colorbar(scatter, ax=ax, label="Density")

        # Add labels and title
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z (θ')")
        ax.set_title("Geometric Projection (3D Topology)", fontsize=14)
        ax.legend()

        # Save plot
        plt.savefig("geometric_projection.png", dpi=300, bbox_inches="tight")
        plt.close()

        # Return metrics for meta-analysis
        return {
            'n': self.numbers,
            'x': coords[:, 0],
            'y': coords[:, 1],
            'z': coords[:, 2],
            'density_score': density,
            'avg_proj_coord': np.mean(coords, axis=1)
        }

    def plot_invariant_frame_map(self, gap_data, zeta_data, geom_data):
        """
        Plot 4: Invariant Frame Map (Meta-Plot)
        Combines data from the three previous plots to create a meta-analysis.
        """
        # Prepare data for common numbers
        common_n = sorted(set(gap_data['n']).intersection(zeta_data['n']).intersection(geom_data['n']))
        common_n = np.array([n for n in common_n if n <= self.max_n])

        # Extract metrics for common numbers
        def extract_for_common(data_dict, key, common_indices):
            if key in data_dict:
                # Handle case where data is only for a subset (like primes)
                if len(data_dict['n']) != len(common_indices):
                    # Interpolate to match common_indices
                    return np.interp(common_indices, data_dict['n'], data_dict[key])
                else:
                    # Direct extraction for matching arrays
                    return data_dict[key]
            return np.ones_like(common_indices)  # Default if key not found

        # Get gap velocity (for primes only)
        gap_velocity = extract_for_common(gap_data, 'gap_velocity', common_n)

        # Get Z values and shift ratios
        z_values = extract_for_common(zeta_data, 'z_value', common_n)
        shift_ratios = extract_for_common(zeta_data, 'shift_ratio', common_n)

        # Get density scores and projection coordinates
        density_scores = extract_for_common(geom_data, 'density_score', common_n)
        avg_proj_coords = extract_for_common(geom_data, 'avg_proj_coord', common_n)

        # Compute meta-score using geometric mean of normalized metrics
        # Replace any zeros with small values to avoid zero products
        gap_velocity_norm = np.maximum(gap_velocity, 1e-10) / np.maximum(np.max(gap_velocity), 1e-10)
        shift_ratios_norm = np.maximum(shift_ratios, 1e-10) / np.maximum(np.max(shift_ratios), 1e-10)
        density_scores_norm = np.maximum(density_scores, 1e-10) / np.maximum(np.max(density_scores), 1e-10)

        meta_score = (gap_velocity_norm * shift_ratios_norm * density_scores_norm) ** (1/3)

        # Apply Lorentz-like transform to z-coordinate
        lorentz_z = self.lorentz_transform(gap_velocity_norm, avg_proj_coords)

        # Create 3D plot
        fig = plt.figure(figsize=(14, 12))
        ax = fig.add_subplot(111, projection='3d')

        # Plot all numbers with meta-score as color and size
        scatter = ax.scatter(
            gap_velocity_norm,
            z_values,
            lorentz_z,
            c=meta_score,
            cmap='plasma',
            s=meta_score * 50 + 5,  # Size based on meta-score
            alpha=0.7,
        )

        # Identify high meta-score numbers (potential primes or special numbers)
        threshold = np.percentile(meta_score, 85)
        high_score_indices = meta_score > threshold
        high_score_numbers = common_n[high_score_indices]

        # Highlight high meta-score numbers
        ax.scatter(
            gap_velocity_norm[high_score_indices],
            z_values[high_score_indices],
            lorentz_z[high_score_indices],
            color="red",
            marker="*",
            s=200,
            edgecolor="white",
            label=f"High Meta-Score (>{threshold:.2f})",
        )

        # Add color bar
        plt.colorbar(scatter, ax=ax, label="Meta-Score")

        # Add labels and title
        ax.set_xlabel("Gap Velocity (normalized)")
        ax.set_ylabel("Z(n) Value")
        ax.set_zlabel("Lorentz-Transformed Coordinate")
        ax.set_title("Invariant Frame Map (Meta-Analysis)", fontsize=14)
        ax.legend()

        # Save plot
        plt.savefig("invariant_frame_map.png", dpi=300, bbox_inches="tight")
        plt.close()

        # Validate high meta-score numbers against known primes
        high_score_primes = [n for n in high_score_numbers if isprime(n)]
        precision = len(high_score_primes) / len(high_score_numbers) if len(high_score_numbers) > 0 else 0

        print(f"\nValidation Results:")
        print(f"High meta-score numbers: {len(high_score_numbers)}")
        print(f"High meta-score primes: {len(high_score_primes)}")
        print(f"Precision: {precision:.2%}")
        print(f"Top 10 high meta-score numbers: {high_score_numbers[:10]}")

        return {
            'meta_score': meta_score,
            'high_score_numbers': high_score_numbers,
            'precision': precision
        }

    def run_analysis(self):
        """
        Run the complete analysis and generate all plots.
        """
        print(f"Starting Z-Transform analysis for numbers up to {self.max_n}...")
        print(f"Found {len(self.primes)} primes in the range.")

        # Generate the three base plots and collect their data
        print("\nGenerating Gap Deviation plot...")
        gap_data = self.plot_gap_deviation()

        print("Generating Zeta Shift plot...")
        zeta_data = self.plot_zeta_shift()

        print("Generating Geometric Projection plot...")
        geom_data = self.plot_geometric_projection()

        # Generate the meta-plot using data from the three base plots
        print("\nGenerating Invariant Frame Map (meta-plot)...")
        meta_data = self.plot_invariant_frame_map(gap_data, zeta_data, geom_data)

        print("\nAnalysis complete! All plots saved.")
        return {
            'gap_data': gap_data,
            'zeta_data': zeta_data,
            'geom_data': geom_data,
            'meta_data': meta_data
        }


if __name__ == "__main__":
    # Create Z-Transform analyzer and run analysis
    analyzer = ZTransform(max_n=N_MAX)
    results = analyzer.run_analysis()