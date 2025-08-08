#!/usr/bin/env python3
"""
Enhanced Spectral Form Factor K(τ)/N Analysis with k* Parameter Sweep
======================================================================

Implements the spectral form factor K(τ)/N computation over both τ and k* dimensions
with bootstrap bands ≈0.05/N for regime-dependent correlation analysis.

This extends the existing spectral_form_factor_analysis.py to include:
1. Two-dimensional analysis over (τ, k*) parameter space
2. Bootstrap uncertainty estimation for regime-dependent correlations
3. Enhanced documentation and visualization
4. CSV outputs for interpretation in Z framework context

Key Features:
- K(τ)/N computation across τ ∈ [0, 10] and k* ∈ [0.1, 0.5]
- Bootstrap confidence bands ≈0.05/N for uncertainty quantification
- Regime-dependent correlation analysis
- Integration with DiscreteZetaShift and Z framework
- Comprehensive plots and CSV outputs

Author: Z Framework / Task 6 Implementation
"""

import numpy as np
import pandas as pd
import mpmath as mp
from scipy.fft import fft, fftfreq
from scipy.stats import entropy, kstest
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from core.domain import DiscreteZetaShift
from core.axioms import universal_invariance, curvature, theta_prime
import time
import warnings
import os
from pathlib import Path

warnings.filterwarnings('ignore')

# Set high precision for zeta computations
mp.mp.dps = 50

# Mathematical constants
PHI = float(mp.phi)
PI = float(mp.pi)
E = float(mp.e)
E_SQUARED = float(mp.exp(2))

class EnhancedSpectralKAnalysis:
    """
    Enhanced spectral form factor analysis with k* parameter sweep for regime-dependent correlations.
    
    This class extends spectral form factor computation to analyze both τ (time/frequency) and 
    k* (curvature exponent) dimensions, providing comprehensive insight into regime-dependent
    correlations within the Z framework.
    """
    
    def __init__(self, M=1000, N=100000, tau_max=10.0, tau_steps=100, 
                 k_min=0.1, k_max=0.5, k_steps=50):
        """
        Initialize enhanced spectral analysis parameters.
        
        Parameters:
        -----------
        M : int
            Number of zeta zeros to compute (up to t=1000+)
        N : int  
            Maximum sequence length for analysis
        tau_max : float
            Maximum τ value for spectral form factor
        tau_steps : int
            Number of τ steps for analysis
        k_min, k_max : float
            Range of k* (curvature exponent) values
        k_steps : int
            Number of k* steps for analysis
        """
        self.M = M
        self.N = N
        self.tau_max = tau_max
        self.tau_steps = tau_steps
        self.k_min = k_min
        self.k_max = k_max
        self.k_steps = k_steps
        
        # Parameter grids
        self.tau_values = np.linspace(0, tau_max, tau_steps)
        self.k_values = np.linspace(k_min, k_max, k_steps)
        self.tau_grid, self.k_grid = np.meshgrid(self.tau_values, self.k_values)
        
        # Storage for results
        self.zeta_zeros = []
        self.unfolded_zeros = []
        self.spectral_form_factor_2d = np.zeros((k_steps, tau_steps))  # K(τ,k*)/N
        self.bootstrap_bands_2d = {'low': np.zeros((k_steps, tau_steps)), 
                                   'high': np.zeros((k_steps, tau_steps))}
        self.regime_correlations = []
        self.k_optimal = 0.200  # From existing analysis
        
        print(f"Initialized enhanced analysis:")
        print(f"  - Zeta zeros: M={M}")
        print(f"  - τ range: [0, {tau_max}] with {tau_steps} steps")
        print(f"  - k* range: [{k_min}, {k_max}] with {k_steps} steps")
        print(f"  - Total parameter combinations: {tau_steps * k_steps}")
        
    def compute_zeta_zeros_with_k_transform(self):
        """
        Compute zeta zeros and apply k*-dependent transformations.
        
        This extends the standard zeta zero computation to include transformations
        based on the curvature parameter k*, providing the foundation for 
        regime-dependent correlation analysis.
        """
        print(f"Computing {self.M} zeta zeros with k*-transforms...")
        start_time = time.time()
        
        self.zeta_zeros = []
        
        for j in range(1, self.M + 1):
            if j % 200 == 0:
                print(f"  Computed {j}/{self.M} zeros")
            
            # Use mpmath to compute zeta zeros
            zero = mp.zetazero(j)
            self.zeta_zeros.append(float(zero.imag))
        
        self.zeta_zeros = np.array(self.zeta_zeros)
        elapsed = time.time() - start_time
        print(f"Computed {len(self.zeta_zeros)} zeros in {elapsed:.2f}s")
        print(f"Zero range: [{self.zeta_zeros[0]:.3f}, {self.zeta_zeros[-1]:.3f}]")
        
        # Apply golden ratio transformation for different k* values
        self.transformed_zeros = {}
        for k in self.k_values:
            # Transform zeros using θ'(t,k) = φ * ((t mod φ)/φ)^k
            t_mod_phi = self.zeta_zeros % PHI
            normalized = t_mod_phi / PHI
            transformed = PHI * (normalized ** k)
            self.transformed_zeros[k] = transformed
        
        print(f"Applied k*-transformations for {len(self.k_values)} k* values")
        return self.zeta_zeros
    
    def unfold_zeros_with_k_dependence(self, k_value):
        """
        Unfold zeta zeros with k*-dependent unfolding for regime analysis.
        
        Parameters:
        -----------
        k_value : float
            Curvature exponent for k*-dependent unfolding
        """
        if len(self.zeta_zeros) == 0:
            self.compute_zeta_zeros_with_k_transform()
        
        # Use k*-transformed zeros for unfolding
        zeros_k = self.transformed_zeros[k_value]
        
        # Riemann-von Mangoldt formula with k*-correction
        def N_riemann_k(t, k):
            """Average number of zeros up to height t with k*-correction"""
            base = (t / (2 * PI)) * np.log(t / (2 * PI * E))
            # k*-dependent correction factor
            k_correction = (k / self.k_optimal) ** 0.5
            return base * k_correction
        
        # Unfold by removing k*-dependent secular trend
        unfolded_k = []
        for i, t in enumerate(zeros_k):
            n_avg = N_riemann_k(t, k_value)
            unfolded = i + 1 - n_avg
            unfolded_k.append(unfolded)
        
        return np.array(unfolded_k)
    
    def compute_spectral_form_factor_2d(self):
        """
        Compute K(τ,k*)/N across the full (τ, k*) parameter space.
        
        This is the core computation that extends the standard spectral form factor
        to include k*-dependence, revealing regime-dependent correlations.
        """
        print("Computing 2D spectral form factor K(τ,k*)/N...")
        print(f"Parameter space: {self.tau_steps} × {self.k_steps} = {self.tau_steps * self.k_steps} points")
        
        total_computations = 0
        start_time = time.time()
        
        for k_idx, k_val in enumerate(self.k_values):
            if k_idx % 10 == 0:
                print(f"  k* = {k_val:.3f} ({k_idx+1}/{len(self.k_values)})")
            
            # Get k*-unfolded zeros for this k* value
            unfolded_zeros_k = self.unfold_zeros_with_k_dependence(k_val)
            N_k = len(unfolded_zeros_k)
            
            for tau_idx, tau_val in enumerate(self.tau_values):
                # Compute K(τ,k*) = |sum_j exp(i*τ*t_j^(k*))|² - N_k
                phase_sum = np.sum(np.exp(1j * tau_val * unfolded_zeros_k))
                K_tau_k = abs(phase_sum)**2 - N_k
                
                # Normalize by N_k
                self.spectral_form_factor_2d[k_idx, tau_idx] = K_tau_k / N_k
                total_computations += 1
        
        elapsed = time.time() - start_time
        print(f"Computed {total_computations} K(τ,k*)/N values in {elapsed:.2f}s")
        print(f"Average: {elapsed/total_computations*1000:.2f}ms per computation")
        
        return self.spectral_form_factor_2d
    
    def compute_bootstrap_bands_2d(self, n_bootstrap=500):
        """
        Compute 2D bootstrap confidence bands for regime-dependent correlations.
        
        This provides uncertainty quantification across the (τ, k*) parameter space,
        with bands approximately ≈0.05/N as specified in the requirements.
        """
        print(f"Computing 2D bootstrap bands with {n_bootstrap} samples...")
        
        bootstrap_results = np.zeros((n_bootstrap, self.k_steps, self.tau_steps))
        
        for bootstrap_idx in range(n_bootstrap):
            if bootstrap_idx % 100 == 0:
                print(f"  Bootstrap sample {bootstrap_idx+1}/{n_bootstrap}")
            
            for k_idx, k_val in enumerate(self.k_values):
                # Generate random levels with k*-dependent statistics
                N_k = len(self.zeta_zeros)  # Use consistent N
                
                # GUE-like spacing with k*-modulation
                base_spacing = np.random.exponential(scale=1.0, size=N_k)
                k_modulation = (k_val / self.k_optimal) ** 0.3  # Moderate k*-dependence
                random_spacings = base_spacing * k_modulation
                random_levels = np.cumsum(random_spacings)
                
                # Compute K(τ,k*) for all τ values
                for tau_idx, tau_val in enumerate(self.tau_values):
                    phase_sum = np.sum(np.exp(1j * tau_val * random_levels))
                    K_tau_k = abs(phase_sum)**2 - N_k
                    bootstrap_results[bootstrap_idx, k_idx, tau_idx] = K_tau_k / N_k
        
        # Compute confidence bands (5th and 95th percentiles)
        self.bootstrap_bands_2d['low'] = np.percentile(bootstrap_results, 5, axis=0)
        self.bootstrap_bands_2d['high'] = np.percentile(bootstrap_results, 95, axis=0)
        
        # Verify ≈0.05/N scaling
        typical_band_width = np.mean(self.bootstrap_bands_2d['high'] - self.bootstrap_bands_2d['low'])
        expected_width = 0.05 / len(self.zeta_zeros)
        print(f"Bootstrap band analysis:")
        print(f"  Typical band width: {typical_band_width:.6f}")
        print(f"  Expected ≈0.05/N: {expected_width:.6f}")
        print(f"  Ratio: {typical_band_width/expected_width:.2f}")
        
        return self.bootstrap_bands_2d
    
    def analyze_regime_dependent_correlations(self):
        """
        Analyze regime-dependent correlations in the (τ, k*) space.
        
        This identifies different correlation regimes and their dependencies on
        both τ and k* parameters, providing insight into the Z framework's
        regime-dependent behavior.
        """
        print("Analyzing regime-dependent correlations...")
        
        # Define regime boundaries based on τ and k* values
        tau_regimes = {
            'low_freq': (0, self.tau_max/3),
            'mid_freq': (self.tau_max/3, 2*self.tau_max/3), 
            'high_freq': (2*self.tau_max/3, self.tau_max)
        }
        
        k_regimes = {
            'low_curve': (self.k_min, self.k_min + (self.k_max-self.k_min)/3),
            'optimal': (self.k_optimal - 0.05, self.k_optimal + 0.05),
            'high_curve': (2*(self.k_max-self.k_min)/3 + self.k_min, self.k_max)
        }
        
        self.regime_correlations = []
        
        for tau_regime, (tau_low, tau_high) in tau_regimes.items():
            for k_regime, (k_low, k_high) in k_regimes.items():
                # Extract data in this regime
                tau_mask = (self.tau_values >= tau_low) & (self.tau_values <= tau_high)
                k_mask = (self.k_values >= k_low) & (self.k_values <= k_high)
                
                # Get indices
                tau_indices = np.where(tau_mask)[0]
                k_indices = np.where(k_mask)[0]
                
                if len(tau_indices) > 0 and len(k_indices) > 0:
                    # Extract regime data
                    regime_data = self.spectral_form_factor_2d[np.ix_(k_indices, tau_indices)]
                    regime_bands_low = self.bootstrap_bands_2d['low'][np.ix_(k_indices, tau_indices)]
                    regime_bands_high = self.bootstrap_bands_2d['high'][np.ix_(k_indices, tau_indices)]
                    
                    # Compute regime statistics
                    mean_correlation = np.mean(regime_data)
                    std_correlation = np.std(regime_data)
                    max_correlation = np.max(regime_data)
                    min_correlation = np.min(regime_data)
                    
                    # Uncertainty measures
                    mean_uncertainty = np.mean(regime_bands_high - regime_bands_low)
                    relative_uncertainty = mean_uncertainty / abs(mean_correlation) if mean_correlation != 0 else np.inf
                    
                    # Correlation strength (deviation from bootstrap baseline)
                    correlation_strength = abs(mean_correlation) / mean_uncertainty if mean_uncertainty > 0 else 0
                    
                    regime_info = {
                        'tau_regime': tau_regime,
                        'k_regime': k_regime,
                        'tau_range': (tau_low, tau_high),
                        'k_range': (k_low, k_high),
                        'mean_correlation': mean_correlation,
                        'std_correlation': std_correlation,
                        'max_correlation': max_correlation,
                        'min_correlation': min_correlation,
                        'mean_uncertainty': mean_uncertainty,
                        'relative_uncertainty': relative_uncertainty,
                        'correlation_strength': correlation_strength,
                        'regime_size': regime_data.shape
                    }
                    
                    self.regime_correlations.append(regime_info)
        
        # Sort by correlation strength
        self.regime_correlations.sort(key=lambda x: x['correlation_strength'], reverse=True)
        
        print(f"Identified {len(self.regime_correlations)} correlation regimes")
        print("\nTop 3 strongest correlation regimes:")
        for i, regime in enumerate(self.regime_correlations[:3]):
            print(f"  {i+1}. τ:{regime['tau_regime']}, k*:{regime['k_regime']}")
            print(f"     Correlation: {regime['mean_correlation']:.4f} ± {regime['mean_uncertainty']:.4f}")
            print(f"     Strength: {regime['correlation_strength']:.2f}")
        
        return self.regime_correlations
    
    def save_results(self, output_dir="enhanced_spectral_k_analysis"):
        """
        Save comprehensive results including 2D analysis and regime correlations.
        """
        os.makedirs(output_dir, exist_ok=True)
        print(f"Saving enhanced results to {output_dir}/")
        
        # 1. 2D Spectral form factor CSV: [τ, k*, K_tau_k, band_low, band_high]
        data_rows = []
        for k_idx, k_val in enumerate(self.k_values):
            for tau_idx, tau_val in enumerate(self.tau_values):
                data_rows.append({
                    'tau': tau_val,
                    'k_star': k_val, 
                    'K_tau_k': self.spectral_form_factor_2d[k_idx, tau_idx],
                    'band_low': self.bootstrap_bands_2d['low'][k_idx, tau_idx],
                    'band_high': self.bootstrap_bands_2d['high'][k_idx, tau_idx],
                    'uncertainty': self.bootstrap_bands_2d['high'][k_idx, tau_idx] - 
                                 self.bootstrap_bands_2d['low'][k_idx, tau_idx]
                })
        
        spectral_2d_df = pd.DataFrame(data_rows)
        spectral_2d_csv = f"{output_dir}/spectral_form_factor_2d.csv"
        spectral_2d_df.to_csv(spectral_2d_csv, index=False)
        print(f"Saved 2D spectral form factor to {spectral_2d_csv}")
        
        # 2. Regime correlation analysis
        if self.regime_correlations:
            regime_df = pd.DataFrame(self.regime_correlations)
            regime_csv = f"{output_dir}/regime_correlations.csv"
            regime_df.to_csv(regime_csv, index=False)
            print(f"Saved regime correlations to {regime_csv}")
        
        # 3. Parameter grids for reference
        params_df = pd.DataFrame({
            'tau_values': self.tau_values,
            'tau_index': range(len(self.tau_values))
        })
        k_params_df = pd.DataFrame({
            'k_values': self.k_values,
            'k_index': range(len(self.k_values))
        })
        
        params_df.to_csv(f"{output_dir}/tau_parameters.csv", index=False)
        k_params_df.to_csv(f"{output_dir}/k_parameters.csv", index=False)
        
        print(f"Saved parameter grids to {output_dir}/")
        return output_dir
    
    def plot_enhanced_results(self, save_plots=True, output_dir="enhanced_spectral_k_analysis"):
        """
        Generate comprehensive 2D plots and regime analysis visualizations.
        """
        print("Generating enhanced 2D plots...")
        
        # Set up the figure with subplots
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Main 2D heatmap of K(τ,k*)/N
        ax1 = plt.subplot(2, 3, 1)
        im1 = ax1.imshow(self.spectral_form_factor_2d, 
                        extent=[0, self.tau_max, self.k_min, self.k_max],
                        aspect='auto', origin='lower', cmap='RdBu_r')
        ax1.set_xlabel('τ')
        ax1.set_ylabel('k*')
        ax1.set_title('Spectral Form Factor K(τ,k*)/N')
        plt.colorbar(im1, ax=ax1, label='K(τ,k*)/N')
        
        # Mark optimal k*
        ax1.axhline(y=self.k_optimal, color='yellow', linestyle='--', linewidth=2, 
                   label=f'k*_optimal = {self.k_optimal}')
        ax1.legend()
        
        # 2. Bootstrap uncertainty map
        ax2 = plt.subplot(2, 3, 2)
        uncertainty_map = self.bootstrap_bands_2d['high'] - self.bootstrap_bands_2d['low']
        im2 = ax2.imshow(uncertainty_map,
                        extent=[0, self.tau_max, self.k_min, self.k_max],
                        aspect='auto', origin='lower', cmap='plasma')
        ax2.set_xlabel('τ')
        ax2.set_ylabel('k*')
        ax2.set_title('Bootstrap Uncertainty (≈0.05/N)')
        plt.colorbar(im2, ax=ax2, label='Uncertainty')
        
        # 3. Signal-to-noise ratio
        ax3 = plt.subplot(2, 3, 3)
        snr_map = np.abs(self.spectral_form_factor_2d) / (uncertainty_map + 1e-10)
        im3 = ax3.imshow(snr_map,
                        extent=[0, self.tau_max, self.k_min, self.k_max],
                        aspect='auto', origin='lower', cmap='viridis')
        ax3.set_xlabel('τ')
        ax3.set_ylabel('k*')
        ax3.set_title('Signal-to-Noise Ratio')
        plt.colorbar(im3, ax=ax3, label='SNR')
        
        # 4. Cross-section at optimal k*
        ax4 = plt.subplot(2, 3, 4)
        k_opt_idx = np.argmin(np.abs(self.k_values - self.k_optimal))
        k_opt_data = self.spectral_form_factor_2d[k_opt_idx, :]
        k_opt_low = self.bootstrap_bands_2d['low'][k_opt_idx, :]
        k_opt_high = self.bootstrap_bands_2d['high'][k_opt_idx, :]
        
        ax4.plot(self.tau_values, k_opt_data, 'b-', linewidth=2, 
                label=f'K(τ,k*={self.k_optimal})/N')
        ax4.fill_between(self.tau_values, k_opt_low, k_opt_high, 
                        alpha=0.3, label='Bootstrap bands')
        ax4.set_xlabel('τ')
        ax4.set_ylabel('K(τ)/N')
        ax4.set_title(f'Cross-section at k* = {self.k_optimal}')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. τ-averaged K vs k*
        ax5 = plt.subplot(2, 3, 5)
        k_averaged = np.mean(self.spectral_form_factor_2d, axis=1)
        k_uncertainty = np.mean(uncertainty_map, axis=1)
        
        ax5.plot(self.k_values, k_averaged, 'r-', linewidth=2, label='⟨K(τ,k*)⟩_τ/N')
        ax5.fill_between(self.k_values, k_averaged - k_uncertainty, k_averaged + k_uncertainty,
                        alpha=0.3, label='Average uncertainty')
        ax5.axvline(x=self.k_optimal, color='yellow', linestyle='--', linewidth=2,
                   label=f'k*_optimal = {self.k_optimal}')
        ax5.set_xlabel('k*')
        ax5.set_ylabel('⟨K(τ,k*)⟩_τ/N')
        ax5.set_title('τ-averaged Spectral Form Factor')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Regime correlation strength map
        ax6 = plt.subplot(2, 3, 6)
        if self.regime_correlations:
            # Create regime strength heatmap
            regime_strength_map = np.zeros_like(self.spectral_form_factor_2d)
            
            for regime in self.regime_correlations:
                tau_low, tau_high = regime['tau_range'] 
                k_low, k_high = regime['k_range']
                
                tau_mask = (self.tau_values >= tau_low) & (self.tau_values <= tau_high)
                k_mask = (self.k_values >= k_low) & (self.k_values <= k_high)
                
                tau_indices = np.where(tau_mask)[0]
                k_indices = np.where(k_mask)[0]
                
                if len(tau_indices) > 0 and len(k_indices) > 0:
                    for k_idx in k_indices:
                        for tau_idx in tau_indices:
                            regime_strength_map[k_idx, tau_idx] = regime['correlation_strength']
            
            im6 = ax6.imshow(regime_strength_map,
                            extent=[0, self.tau_max, self.k_min, self.k_max],
                            aspect='auto', origin='lower', cmap='hot')
            ax6.set_xlabel('τ')
            ax6.set_ylabel('k*')
            ax6.set_title('Regime Correlation Strength')
            plt.colorbar(im6, ax=ax6, label='Correlation Strength')
        
        plt.tight_layout()
        
        if save_plots:
            os.makedirs(output_dir, exist_ok=True)
            plot_file = f"{output_dir}/enhanced_spectral_k_analysis.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            print(f"Saved enhanced plots to {plot_file}")
        
        return fig
    
    def run_complete_enhanced_analysis(self, save_results=True, plot_results=True):
        """
        Run the complete enhanced spectral form factor analysis with k* sweep.
        """
        print("="*80)
        print("ENHANCED SPECTRAL FORM FACTOR K(τ,k*)/N ANALYSIS")
        print("Regime-Dependent Correlations with Bootstrap Bands")
        print("="*80)
        
        start_time = time.time()
        
        # Step 1: Compute zeta zeros with k*-transforms
        self.compute_zeta_zeros_with_k_transform()
        
        # Step 2: Compute 2D spectral form factor
        self.compute_spectral_form_factor_2d()
        
        # Step 3: Bootstrap confidence bands
        self.compute_bootstrap_bands_2d(n_bootstrap=200)  # Reduced for faster execution
        
        # Step 4: Regime-dependent correlation analysis
        self.analyze_regime_dependent_correlations()
        
        # Step 5: Save results
        if save_results:
            output_dir = self.save_results()
        
        # Step 6: Generate plots
        if plot_results:
            self.plot_enhanced_results(save_plots=save_results)
        
        total_time = time.time() - start_time
        
        print("="*80)
        print("ENHANCED ANALYSIS COMPLETE")
        print("="*80)
        print(f"Total runtime: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
        print(f"Zeta zeros computed: {len(self.zeta_zeros)}")
        print(f"2D parameter space: {self.k_steps} × {self.tau_steps} = {self.k_steps * self.tau_steps} points")
        print(f"Correlation regimes identified: {len(self.regime_correlations)}")
        print(f"Bootstrap samples per point: 200")
        
        # Key findings summary
        if self.regime_correlations:
            strongest_regime = self.regime_correlations[0]
            print(f"\nKey Findings:")
            print(f"- Strongest correlation in {strongest_regime['tau_regime']} τ, {strongest_regime['k_regime']} k* regime")
            print(f"- Correlation strength: {strongest_regime['correlation_strength']:.2f}")
            print(f"- Mean correlation: {strongest_regime['mean_correlation']:.4f}")
            print(f"- Relative uncertainty: {strongest_regime['relative_uncertainty']:.2%}")
        
        return {
            'zeta_zeros': self.zeta_zeros,
            'spectral_form_factor_2d': self.spectral_form_factor_2d,
            'bootstrap_bands_2d': self.bootstrap_bands_2d,
            'regime_correlations': self.regime_correlations,
            'runtime': total_time,
            'parameter_space_size': self.k_steps * self.tau_steps
        }


def main():
    """
    Main execution function for enhanced spectral K(τ,k*) analysis.
    """
    # Initialize enhanced analysis with moderate parameters for testing
    analysis = EnhancedSpectralKAnalysis(
        M=500,           # Zeta zeros (reduced for faster testing)
        N=50000,         # Analysis range
        tau_max=10.0,    # τ range [0,10] as specified
        tau_steps=50,    # 50 τ points (reduced for testing)
        k_min=0.1,       # k* range around optimal
        k_max=0.5,
        k_steps=25       # 25 k* points (reduced for testing)
    )
    
    # Run complete enhanced analysis
    results = analysis.run_complete_enhanced_analysis(
        save_results=True,
        plot_results=True
    )
    
    return results


if __name__ == "__main__":
    # Set matplotlib backend for headless environment
    plt.switch_backend('Agg')
    
    print("Starting Enhanced Spectral Form Factor K(τ,k*)/N Analysis")
    print("Task 6: Regime-Dependent Correlations with Bootstrap Bands")
    results = main()
    print("Enhanced analysis completed successfully!")