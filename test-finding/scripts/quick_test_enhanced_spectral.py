#!/usr/bin/env python3
"""
Quick Test for Enhanced Spectral Form Factor K(τ,k*)/N Analysis
===============================================================

Rapid validation test with minimal parameters to verify implementation.
"""

import sys
import os
sys.path.append('/home/runner/work/unified-framework/unified-framework/test-finding/scripts')

from enhanced_spectral_k_analysis import EnhancedSpectralKAnalysis
import matplotlib.pyplot as plt
plt.switch_backend('Agg')

def quick_test():
    """Run quick validation test with minimal parameters."""
    
    print("="*60)
    print("QUICK TEST: Enhanced Spectral K(τ,k*) Analysis")
    print("="*60)
    
    # Minimal parameters for fast testing
    analysis = EnhancedSpectralKAnalysis(
        M=50,            # Just 50 zeta zeros
        N=1000,          # Small range
        tau_max=5.0,     # Reduced τ range
        tau_steps=10,    # 10 τ points
        k_min=0.18,      # Narrow k* range around optimal
        k_max=0.22,
        k_steps=5        # 5 k* points  
    )
    
    print(f"Test configuration:")
    print(f"  - Parameter space: {analysis.k_steps} × {analysis.tau_steps} = {analysis.k_steps * analysis.tau_steps} points")
    print(f"  - Zeta zeros: {analysis.M}")
    print(f"  - Expected runtime: ~30-60 seconds")
    
    # Run analysis with minimal bootstrap
    try:
        print("\n1. Computing zeta zeros...")
        analysis.compute_zeta_zeros_with_k_transform()
        
        print("2. Computing 2D spectral form factor...")
        analysis.compute_spectral_form_factor_2d()
        
        print("3. Computing bootstrap bands (50 samples)...")
        analysis.compute_bootstrap_bands_2d(n_bootstrap=50)
        
        print("4. Analyzing regimes...")
        analysis.analyze_regime_dependent_correlations()
        
        print("5. Saving results...")
        output_dir = analysis.save_results(output_dir="quick_test_results")
        
        print("6. Generating plots...")
        analysis.plot_enhanced_results(save_plots=True, output_dir=output_dir)
        
        print("\n" + "="*60)
        print("QUICK TEST COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        # Summary
        print(f"Results saved to: {output_dir}")
        print(f"Zeta zeros computed: {len(analysis.zeta_zeros)}")
        print(f"Regimes identified: {len(analysis.regime_correlations)}")
        
        if analysis.regime_correlations:
            strongest = analysis.regime_correlations[0]
            print(f"Strongest correlation: {strongest['tau_regime']} τ, {strongest['k_regime']} k*")
            print(f"Correlation strength: {strongest['correlation_strength']:.2f}")
        
        # Check file sizes
        import os
        files = os.listdir(output_dir)
        print(f"\nGenerated files:")
        for file in files:
            filepath = os.path.join(output_dir, file)
            size_kb = os.path.getsize(filepath) / 1024
            print(f"  - {file}: {size_kb:.1f} KB")
        
        return True
        
    except Exception as e:
        print(f"\nQUICK TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = quick_test()
    print(f"\nTest result: {'PASS' if success else 'FAIL'}")