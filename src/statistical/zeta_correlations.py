"""
Statistical validation framework for Z framework correlations.

This module implements statistical validation functions for correlating prime geodesics
with Riemann zeta zero embeddings, targeting Pearson r ≥ 0.95 and KS statistic ≥ 0.92.

Integrates with existing DiscreteZetaShift framework using high-precision mpmath
computations and 5D helical embeddings for geometric analysis.
"""

import numpy as np
import mpmath as mp
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.stats import kstest
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from core.domain import DiscreteZetaShift
from core.axioms import theta_prime, curvature

# High precision settings
mp.mp.dps = 50
PHI = (1 + mp.sqrt(5)) / 2
E_SQUARED = mp.exp(2)

class ZetaCorrelationAnalyzer:
    """
    Statistical analyzer for prime-zeta zero correlations using GMM and geodesic analysis.
    
    Implements validation targeting:
    - Pearson correlation r ≥ 0.95 between prime geodesics and zeta zero spacings
    - KS statistic ≥ 0.92 for distribution similarity
    - GMM clustering with BIC optimization for pattern recognition
    """
    
    def __init__(self, precision_dps=50):
        """Initialize analyzer with high precision settings."""
        mp.mp.dps = precision_dps
        self.scaler = StandardScaler()
        self.gmm_cache = {}
        
    def generate_prime_geodesics(self, N_max=10000, k_optimal=0.200):
        """
        Generate prime geodesic coordinates using θ'(n,k) transformation.
        
        Args:
            N_max (int): Maximum n value for prime generation
            k_optimal (float): Optimal curvature parameter (validated k* ≈ 0.200)
            
        Returns:
            dict: {
                'primes': list of prime numbers,
                'geodesic_coords': 5D coordinates for each prime,
                'theta_prime_values': θ'(p,k) values for each prime,
                'curvature_values': κ(p) values for each prime
            }
        """
        from sympy import isprime
        
        # Generate primes up to N_max
        primes = [p for p in range(2, N_max) if isprime(p)]
        
        geodesic_coords = []
        theta_prime_values = []
        curvature_values = []
        
        for p in primes:
            # Generate DiscreteZetaShift for each prime
            zeta_shift = DiscreteZetaShift(p)
            coords_5d = zeta_shift.get_5d_coordinates()
            geodesic_coords.append(coords_5d)
            
            # Compute θ'(p,k) transformation
            theta_p = float(theta_prime(p, k_optimal))
            theta_prime_values.append(theta_p)
            
            # Compute κ(p) curvature
            from sympy import divisors
            d_p = len(list(divisors(p)))
            kappa_p = float(curvature(p, d_p))
            curvature_values.append(kappa_p)
        
        return {
            'primes': primes,
            'geodesic_coords': geodesic_coords,
            'theta_prime_values': theta_prime_values,
            'curvature_values': curvature_values
        }
    
    def generate_zeta_zeros(self, j_max=1000):
        """
        Generate Riemann zeta zero embeddings using mpmath high precision.
        
        Args:
            j_max (int): Maximum number of zeta zeros to compute
            
        Returns:
            dict: {
                'zeta_zeros': complex values of zeta zeros,
                'zero_spacings': spacings between consecutive zeros,
                'helical_coords': 5D helical embeddings for zeros,
                'unfolded_spacings': processed spacings for correlation
            }
        """
        zeta_zeros = []
        
        # Use mpmath to compute Riemann zeta zeros with high precision
        try:
            for j in range(1, min(j_max + 1, 10000)):  # Limit for computational feasibility
                try:
                    # Compute j-th zeta zero using mpmath
                    zero = mp.zetazero(j)
                    zeta_zeros.append(complex(zero))
                except:
                    # If computation fails, use approximation
                    t_approx = 2 * mp.pi * j / mp.log(j / (2 * mp.pi * mp.e)) if j > 1 else 14.134725
                    zero = complex(0.5, float(t_approx))
                    zeta_zeros.append(zero)
        except Exception as e:
            print(f"Warning: Using approximation for zeta zeros due to: {e}")
            # Fallback to approximation for all zeros
            for j in range(1, j_max + 1):
                if j == 1:
                    t_approx = 14.134725142  # First zero
                else:
                    t_approx = 2 * mp.pi * j / mp.log(j / (2 * mp.pi * mp.e))
                zero = complex(0.5, float(t_approx))
                zeta_zeros.append(zero)
        
        # Compute zero spacings
        zero_spacings = []
        for i in range(1, len(zeta_zeros)):
            spacing = abs(zeta_zeros[i].imag - zeta_zeros[i-1].imag)
            zero_spacings.append(spacing)
        
        # Generate helical coordinates for zeros
        helical_coords = []
        for i, zero in enumerate(zeta_zeros):
            # Map zero to 5D coordinates using similar methodology as DiscreteZetaShift
            n_equivalent = int(abs(zero.imag))
            
            try:
                zeta_shift = DiscreteZetaShift(max(2, n_equivalent))
                coords = zeta_shift.get_5d_coordinates()
                helical_coords.append(coords)
            except:
                # Fallback direct computation
                phi = float(PHI)
                theta_d = phi * ((n_equivalent % phi) / phi) ** 0.3
                theta_e = phi * (((n_equivalent + 1) % phi) / phi) ** 0.3
                x = n_equivalent * mp.cos(theta_d)
                y = n_equivalent * mp.sin(theta_e)
                z = float(zero.real)
                w = float(zero.imag) / phi
                u = (n_equivalent % phi) / phi
                helical_coords.append((float(x), float(y), float(z), float(w), float(u)))
        
        # Unfold spacings for correlation analysis
        unfolded_spacings = self._unfold_zero_spacings(zero_spacings)
        
        return {
            'zeta_zeros': zeta_zeros,
            'zero_spacings': zero_spacings,
            'helical_coords': helical_coords,
            'unfolded_spacings': unfolded_spacings
        }
    
    def _unfold_zero_spacings(self, zero_spacings):
        """
        Unfold zeta zero spacings to remove correlations and enable comparison with primes.
        
        Args:
            zero_spacings (list): Raw spacings between consecutive zeros
            
        Returns:
            list: Unfolded spacings suitable for correlation analysis
        """
        if not zero_spacings:
            return []
        
        # Apply spectral unfolding to remove level repulsion
        mean_spacing = np.mean(zero_spacings)
        
        # Normalize by local density (Weyl's law approximation)
        unfolded = []
        cumulative = 0
        
        for i, spacing in enumerate(zero_spacings):
            # Local density correction using Weyl's asymptotic formula
            t = sum(zero_spacings[:i+1])  # Approximate height
            if t > 0:
                local_density = mp.log(t / (2 * mp.pi)) / (2 * mp.pi)
                normalized_spacing = spacing * float(local_density)
            else:
                normalized_spacing = spacing
            
            cumulative += normalized_spacing
            unfolded.append(cumulative)
        
        # Convert to spacings in unfolded coordinates
        unfolded_spacings = [unfolded[i+1] - unfolded[i] for i in range(len(unfolded)-1)]
        
        return unfolded_spacings
    
    def compute_gmm_correlation(self, prime_data, zeta_data, n_components=5):
        """
        Compute GMM-based correlation between prime geodesics and zeta zero patterns.
        
        Args:
            prime_data (dict): Result from generate_prime_geodesics()
            zeta_data (dict): Result from generate_zeta_zeros()
            n_components (int): Number of GMM components
            
        Returns:
            dict: {
                'gmm_score': BIC-optimized GMM score,
                'cluster_correlation': Correlation between cluster assignments,
                'component_overlap': Overlap between prime and zeta components,
                'validation_passed': True if correlation meets targets
            }
        """
        # Prepare data for GMM analysis
        prime_features = np.array(prime_data['geodesic_coords'])
        zeta_features = np.array(zeta_data['helical_coords'])
        
        # Ensure same dimensionality
        min_samples = min(len(prime_features), len(zeta_features))
        prime_features = prime_features[:min_samples]
        zeta_features = zeta_features[:min_samples]
        
        # Standardize features
        prime_features_scaled = self.scaler.fit_transform(prime_features)
        zeta_features_scaled = self.scaler.fit_transform(zeta_features)
        
        # Fit GMM to prime data
        gmm_prime = GaussianMixture(n_components=n_components, random_state=42)
        prime_labels = gmm_prime.fit_predict(prime_features_scaled)
        
        # Fit GMM to zeta data
        gmm_zeta = GaussianMixture(n_components=n_components, random_state=42)
        zeta_labels = gmm_zeta.fit_predict(zeta_features_scaled)
        
        # Compute correlation between cluster assignments
        cluster_correlation = np.corrcoef(prime_labels, zeta_labels)[0, 1]
        
        # Compute component overlap using mean distances
        component_overlaps = []
        for i in range(n_components):
            prime_mask = prime_labels == i
            zeta_mask = zeta_labels == i
            
            if np.sum(prime_mask) > 0 and np.sum(zeta_mask) > 0:
                prime_centroid = np.mean(prime_features_scaled[prime_mask], axis=0)
                zeta_centroid = np.mean(zeta_features_scaled[zeta_mask], axis=0)
                overlap = 1.0 / (1.0 + np.linalg.norm(prime_centroid - zeta_centroid))
                component_overlaps.append(overlap)
        
        avg_component_overlap = np.mean(component_overlaps) if component_overlaps else 0.0
        
        # Combined GMM score
        gmm_score = 0.5 * abs(cluster_correlation) + 0.5 * avg_component_overlap
        
        # Validation check
        validation_passed = gmm_score >= 0.80  # Reasonable threshold for GMM correlation
        
        return {
            'gmm_score': gmm_score,
            'cluster_correlation': cluster_correlation,
            'component_overlap': avg_component_overlap,
            'n_components_used': n_components,
            'validation_passed': validation_passed,
            'prime_bic': gmm_prime.bic(prime_features_scaled),
            'zeta_bic': gmm_zeta.bic(zeta_features_scaled)
        }
    
    def compute_pearson_correlation(self, prime_data, zeta_data):
        """
        Compute Pearson correlation between prime curvatures and zeta spacings.
        
        Args:
            prime_data (dict): Result from generate_prime_geodesics()
            zeta_data (dict): Result from generate_zeta_zeros()
            
        Returns:
            dict: {
                'pearson_r': Pearson correlation coefficient,
                'p_value': Statistical significance,
                'validation_passed': True if r ≥ 0.95,
                'sample_size': Number of paired samples
            }
        """
        # Extract comparable sequences
        prime_curvatures = prime_data['curvature_values']
        zeta_spacings = zeta_data['unfolded_spacings']
        
        # Ensure same length for correlation
        min_length = min(len(prime_curvatures), len(zeta_spacings))
        if min_length < 10:
            return {
                'pearson_r': 0.0,
                'p_value': 1.0,
                'validation_passed': False,
                'sample_size': min_length,
                'error': 'Insufficient data for correlation'
            }
        
        prime_seq = prime_curvatures[:min_length]
        zeta_seq = zeta_spacings[:min_length]
        
        # Compute Pearson correlation
        pearson_r, p_value = stats.pearsonr(prime_seq, zeta_seq)
        
        # Validation check
        validation_passed = abs(pearson_r) >= 0.95
        
        return {
            'pearson_r': pearson_r,
            'p_value': p_value,
            'validation_passed': validation_passed,
            'sample_size': min_length,
            'prime_mean': np.mean(prime_seq),
            'zeta_mean': np.mean(zeta_seq),
            'prime_std': np.std(prime_seq),
            'zeta_std': np.std(zeta_seq)
        }
    
    def compute_ks_statistic(self, prime_data, zeta_data):
        """
        Compute Kolmogorov-Smirnov statistic for distribution similarity.
        
        Args:
            prime_data (dict): Result from generate_prime_geodesics()
            zeta_data (dict): Result from generate_zeta_zeros()
            
        Returns:
            dict: {
                'ks_statistic': KS test statistic,
                'ks_p_value': P-value for the test,
                'validation_passed': True if KS statistic ≥ 0.92,
                'distribution_similarity': Similarity measure
            }
        """
        # Extract distributions for comparison
        prime_theta = prime_data['theta_prime_values']
        zeta_spacings = zeta_data['zero_spacings']
        
        # Normalize both distributions to [0, 1] for fair comparison
        if len(prime_theta) > 0 and len(zeta_spacings) > 0:
            prime_norm = (np.array(prime_theta) - np.min(prime_theta)) / (np.max(prime_theta) - np.min(prime_theta))
            zeta_norm = (np.array(zeta_spacings) - np.min(zeta_spacings)) / (np.max(zeta_spacings) - np.min(zeta_spacings))
            
            # Ensure same length
            min_length = min(len(prime_norm), len(zeta_norm))
            prime_norm = prime_norm[:min_length]
            zeta_norm = zeta_norm[:min_length]
            
            # Two-sample KS test
            ks_statistic, ks_p_value = stats.ks_2samp(prime_norm, zeta_norm)
            
            # For validation, we want low KS statistic (similar distributions)
            # Convert to similarity measure: similarity = 1 - ks_statistic
            distribution_similarity = 1.0 - ks_statistic
            
            # Validation check (target ≥ 0.92 similarity)
            validation_passed = distribution_similarity >= 0.92
            
            return {
                'ks_statistic': ks_statistic,
                'ks_p_value': ks_p_value,
                'distribution_similarity': distribution_similarity,
                'validation_passed': validation_passed,
                'sample_size': min_length
            }
        else:
            return {
                'ks_statistic': 1.0,
                'ks_p_value': 0.0,
                'distribution_similarity': 0.0,
                'validation_passed': False,
                'sample_size': 0,
                'error': 'Insufficient data for KS test'
            }
    
    def comprehensive_validation(self, N_max=10000, j_max=1000, k_optimal=0.200):
        """
        Run comprehensive validation targeting all statistical requirements.
        
        Args:
            N_max (int): Maximum n for prime generation
            j_max (int): Maximum zeta zeros to compute
            k_optimal (float): Optimal curvature parameter
            
        Returns:
            dict: Complete validation results with pass/fail status
        """
        print("Generating prime geodesics...")
        prime_data = self.generate_prime_geodesics(N_max, k_optimal)
        
        print("Generating zeta zero embeddings...")
        zeta_data = self.generate_zeta_zeros(j_max)
        
        print("Computing GMM correlation...")
        gmm_result = self.compute_gmm_correlation(prime_data, zeta_data)
        
        print("Computing Pearson correlation...")
        pearson_result = self.compute_pearson_correlation(prime_data, zeta_data)
        
        print("Computing KS statistic...")
        ks_result = self.compute_ks_statistic(prime_data, zeta_data)
        
        # Overall validation
        overall_passed = (
            gmm_result['validation_passed'] and
            pearson_result['validation_passed'] and
            ks_result['validation_passed']
        )
        
        return {
            'overall_validation_passed': overall_passed,
            'prime_data_summary': {
                'num_primes': len(prime_data['primes']),
                'max_prime': max(prime_data['primes']) if prime_data['primes'] else 0,
                'mean_curvature': np.mean(prime_data['curvature_values']) if prime_data['curvature_values'] else 0
            },
            'zeta_data_summary': {
                'num_zeros': len(zeta_data['zeta_zeros']),
                'max_zero_height': max([abs(z.imag) for z in zeta_data['zeta_zeros']]) if zeta_data['zeta_zeros'] else 0,
                'mean_spacing': np.mean(zeta_data['zero_spacings']) if zeta_data['zero_spacings'] else 0
            },
            'gmm_validation': gmm_result,
            'pearson_validation': pearson_result,
            'ks_validation': ks_result,
            'parameters_used': {
                'N_max': N_max,
                'j_max': j_max,
                'k_optimal': k_optimal,
                'precision_dps': mp.mp.dps
            }
        }

def validate_z_framework_correlations(N_max=5000, j_max=500, k_optimal=0.200):
    """
    Convenience function for quick Z framework correlation validation.
    
    Args:
        N_max (int): Maximum n for prime analysis
        j_max (int): Maximum zeta zeros
        k_optimal (float): Curvature parameter
        
    Returns:
        dict: Validation results
    """
    analyzer = ZetaCorrelationAnalyzer()
    return analyzer.comprehensive_validation(N_max, j_max, k_optimal)

if __name__ == "__main__":
    # Quick validation test
    print("Running Z framework correlation validation...")
    results = validate_z_framework_correlations(N_max=1000, j_max=100)
    
    print(f"\nValidation Results:")
    print(f"Overall Passed: {results['overall_validation_passed']}")
    print(f"Pearson r: {results['pearson_validation']['pearson_r']:.4f} (target ≥ 0.95)")
    print(f"KS Similarity: {results['ks_validation']['distribution_similarity']:.4f} (target ≥ 0.92)")
    print(f"GMM Score: {results['gmm_validation']['gmm_score']:.4f}")