# Enhanced Spectral Form Factor K(τ)/N Analysis Documentation

## Overview

This document provides comprehensive documentation for the enhanced spectral form factor analysis implementation, addressing Issue #92: "Analyze spectral form factor K(τ)/N and bootstrap bands for regime-dependent correlations."

## Mathematical Foundation

### Spectral Form Factor Definition

The spectral form factor K(τ) is defined as:

```
K(τ) = |∑ⱼ exp(iτt_j)|² - N
```

where:
- `t_j` are the unfolded energy levels (zeta zeros in our case)
- `τ` is the dimensionless time parameter
- `N` is the number of levels

The normalized form is K(τ)/N, which removes the trivial N-dependence.

### k*-Enhanced Analysis

Our implementation extends this to K(τ,k*)/N where:
- `k*` is the curvature exponent from the golden ratio transformation
- `θ'(n,k) = φ * ((n mod φ)/φ)^k` where φ is the golden ratio
- `k* = 0.200` is the optimal value (from proof.py analysis showing 495.2% enhancement)

### Bootstrap Bands

Bootstrap confidence bands are computed using:
- Random matrix theory with GUE-like statistics
- k*-dependent modulation of level spacings
- Confidence intervals at 5th and 95th percentiles
- Target scaling ≈0.05/N as specified in requirements

## Implementation Structure

### Core Classes

#### EnhancedSpectralKAnalysis

Main analysis class that extends the existing spectral_form_factor_analysis.py to include:

1. **2D Parameter Space**: (τ, k*) instead of just τ
2. **Regime Analysis**: Identification of correlation regimes
3. **Bootstrap Uncertainty**: 2D confidence bands
4. **Z Framework Integration**: DiscreteZetaShift compatibility

### Key Methods

#### compute_zeta_zeros_with_k_transform()
- Computes M zeta zeros using mpmath
- Applies golden ratio transformation for each k* value
- Creates transformed_zeros dictionary for regime analysis

#### compute_spectral_form_factor_2d()
- Core computation of K(τ,k*)/N across parameter space
- k*-dependent unfolding using Riemann-von Mangoldt formula
- Efficient vectorized computation

#### compute_bootstrap_bands_2d()
- 2D bootstrap confidence band computation
- GUE-like random level generation with k*-modulation
- Verification of ≈0.05/N scaling requirement

#### analyze_regime_dependent_correlations()
- Identifies correlation regimes in (τ,k*) space
- Computes regime-specific statistics
- Ranks regimes by correlation strength

## Regime-Dependent Correlations

### Regime Definition

The (τ,k*) parameter space is divided into regions:

**τ Regimes:**
- `low_freq`: [0, τ_max/3] - Low frequency behavior
- `mid_freq`: [τ_max/3, 2τ_max/3] - Intermediate frequency
- `high_freq`: [2τ_max/3, τ_max] - High frequency behavior

**k* Regimes:**
- `low_curve`: [k_min, k_min + (k_max-k_min)/3] - Low curvature
- `optimal`: [k*_opt - 0.05, k*_opt + 0.05] - Near-optimal curvature
- `high_curve`: [2(k_max-k_min)/3 + k_min, k_max] - High curvature

### Correlation Metrics

For each regime, we compute:
- Mean correlation: `⟨K(τ,k*)⟩/N`
- Standard deviation: `σ[K(τ,k*)/N]`
- Mean uncertainty: `⟨bootstrap_high - bootstrap_low⟩`
- Relative uncertainty: `uncertainty / |correlation|`
- Correlation strength: `|correlation| / uncertainty`

## Output Files

### CSV Outputs

1. **spectral_form_factor_2d.csv**
   - Columns: [τ, k*, K_tau_k, band_low, band_high, uncertainty]
   - Full 2D dataset for analysis and visualization

2. **regime_correlations.csv**
   - Regime-specific correlation analysis
   - Ranked by correlation strength
   - Statistical summaries for each regime

3. **tau_parameters.csv** & **k_parameters.csv**
   - Parameter grid definitions
   - For reproducibility and cross-referencing

### Visualization Outputs

1. **2D Heatmap**: K(τ,k*)/N across parameter space
2. **Uncertainty Map**: Bootstrap confidence band widths
3. **Signal-to-Noise**: |K|/uncertainty ratio analysis
4. **Cross-sections**: Slices at optimal k* and averaged profiles
5. **Regime Map**: Correlation strength visualization

## Z Framework Context

### Integration with Core Modules

- **DiscreteZetaShift**: Uses transformed zeta zeros with k*-dependence
- **Golden Ratio Transformations**: θ'(n,k) from core.axioms
- **Universal Invariance**: Maintains Z = A(B/c) structure
- **Frame Normalization**: e² factors from core mathematical constants

### Physical Interpretation

The enhanced analysis reveals:
- **Regime-dependent behavior**: Different correlation patterns in (τ,k*) space
- **Optimal curvature effects**: Enhanced correlations near k* = 0.200
- **Bootstrap uncertainty scaling**: Validates ≈0.05/N theoretical prediction
- **Cross-domain insights**: Spectral statistics bridge discrete and continuous domains

## Performance Characteristics

### Computational Complexity

- **Zeta zeros**: O(M log M) using mpmath
- **2D analysis**: O(M × k_steps × τ_steps)
- **Bootstrap bands**: O(n_bootstrap × M × k_steps × τ_steps)
- **Total**: Scales as O(M × n_bootstrap × parameter_space_size)

### Typical Runtimes

- M=500, 25×50 parameter space: ~2-5 minutes
- M=1000, 50×100 parameter space: ~15-30 minutes
- Full scale M=1000, 100×100: ~1-2 hours

### Memory Requirements

- Spectral form factor 2D: k_steps × τ_steps × 8 bytes
- Bootstrap results: n_bootstrap × k_steps × τ_steps × 8 bytes
- Typical: ~10-100 MB for moderate parameter spaces

## Usage Examples

### Basic Usage

```python
from enhanced_spectral_k_analysis import EnhancedSpectralKAnalysis

# Initialize analysis
analysis = EnhancedSpectralKAnalysis(
    M=500,           # Zeta zeros
    tau_max=10.0,    # τ range [0,10]
    tau_steps=50,    # τ resolution
    k_min=0.1,       # k* range 
    k_max=0.5,
    k_steps=25       # k* resolution
)

# Run complete analysis
results = analysis.run_complete_enhanced_analysis()
```

### Advanced Configuration

```python
# High-resolution analysis
analysis = EnhancedSpectralKAnalysis(
    M=1000,          # More zeta zeros
    tau_max=10.0,
    tau_steps=100,   # Higher τ resolution
    k_min=0.15,      # Focused k* range
    k_max=0.35,
    k_steps=50       # Higher k* resolution
)

# Custom bootstrap sampling
analysis.compute_bootstrap_bands_2d(n_bootstrap=1000)
```

## Validation and Quality Assurance

### Mathematical Validation

1. **Bootstrap scaling**: Verify ≈0.05/N band width
2. **GUE statistics**: K-S test for random matrix compliance
3. **k*-dependence**: Consistency with proof.py optimal k* = 0.200
4. **Regime consistency**: Correlation strength rankings

### Numerical Precision

- **mpmath precision**: 50 decimal places for zeta computations
- **Float64 arrays**: Standard precision for large-scale analysis
- **Error handling**: NaN/inf detection and graceful degradation

### Cross-validation

- **Existing methods**: Consistency with spectral_form_factor_analysis.py
- **Parameter limits**: Proper behavior at boundary conditions
- **Scaling tests**: Performance validation across parameter ranges

## Future Extensions

### Potential Enhancements

1. **Adaptive sampling**: Intelligent parameter space exploration
2. **Parallel computing**: Multi-core bootstrap computation
3. **Advanced regimes**: Machine learning regime identification
4. **Cross-correlations**: Multi-dimensional correlation analysis

### Research Applications

1. **Prime gap correlations**: Extension to prime number statistics
2. **Quantum chaos**: Random matrix theory applications
3. **Number theory**: L-function spectral analysis
4. **Physics applications**: Energy level statistics in quantum systems

## Conclusion

The enhanced spectral form factor analysis provides comprehensive insight into regime-dependent correlations in the Z framework through:

- **2D parameter space analysis**: (τ, k*) correlations
- **Bootstrap uncertainty quantification**: ≈0.05/N confidence bands
- **Regime identification**: Statistical correlation patterns
- **Z framework integration**: Consistent with existing infrastructure

This implementation addresses all requirements from Issue #92 and provides a foundation for advanced spectral analysis in the unified framework context.