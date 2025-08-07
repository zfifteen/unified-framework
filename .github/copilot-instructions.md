# Z Framework: Mathematical Research Repository

The Z Framework is a unified mathematical model bridging physical and discrete domains through the empirical invariance of the speed of light. It leverages the universal form Z = A(B/c) to analyze prime number distributions using geometric constraints and curvature-based geodesics.

Always reference these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.

## Working Effectively

### Environment Setup and Dependencies
- Install Python dependencies:
  - `pip3 install numpy pandas matplotlib mpmath sympy scikit-learn statsmodels scipy seaborn plotly`
- Set Python path for imports:
  - `export PYTHONPATH=/home/runner/work/unified-framework/unified-framework`
  - OR prefix commands: `PYTHONPATH=/home/runner/work/unified-framework/unified-framework python3 script.py`

### Core Mathematical Computations
- Test basic framework:
  - `python3 -c "from core.axioms import universal_invariance; print('Test:', universal_invariance(1.0, 3e8))"`
  - Takes: ~0.1 seconds
- Test discrete zeta shift computations:
  - `python3 -c "from core.domain import DiscreteZetaShift; dz = DiscreteZetaShift(10); print('Works')"`
  - Takes: ~0.7 seconds (includes high-precision mpmath initialization)

### Key Computational Scripts
- Run prime curvature proof analysis:
  - `python3 number-theory/prime-curve/proof.py`
  - Takes: ~2 seconds. NEVER CANCEL. Set timeout to 30+ seconds.
  - Computes optimal curvature exponent k* = 0.200 with 495.2% max mid-bin enhancement
- Run hologram visualizations:
  - `python3 number-theory/prime-curve/hologram.py`
  - Takes: ~1.3 seconds
- Run golden curve analysis:
  - `PYTHONPATH=/home/runner/work/unified-framework/unified-framework python3 experiments/lab/golden-curve/brute_force.py`
  - Takes: ~0.8 seconds. Tests Bell inequality violation in prime distributions.
- Run comprehensive data generation:
  - `PYTHONPATH=/home/runner/work/unified-framework/unified-framework python3 applications/test.py`
  - Takes: ~27 seconds. NEVER CANCEL. Set timeout to 60+ minutes for larger datasets.

### Performance Scaling
- 100 DiscreteZetaShift instances: ~0.01 seconds
- 1000 DiscreteZetaShift instances with full computation: ~2 seconds
- Large-scale analysis (applications/test.py): ~27 seconds
- Prime hologram bootstrap (1000 primes): ~0.3 seconds

## Repository Structure

### Core Framework (/core/)
- `axioms.py` - Universal invariance functions, curvature calculations, golden ratio transformations
- `domain.py` - DiscreteZetaShift class with 5D helical embeddings and zeta shift computations
- `orbital.py` - Orbital mechanics and geometric projections

### Applications (/applications/)
- `test.py` - Main comprehensive test suite (27 seconds runtime)
- `helix_hologram.py` - Helical visualization system (~0.01 seconds)
- `load_data.py` - Data loading utilities
- Various visualization and encryption tools

### Research Experiments (/experiments/)
- `/lab/golden-curve/` - Golden ratio curvature analysis
- `/lab/light_primes/` - Prime hologram and density analysis
- `/lab/universal_frame_shift_transformer/` - Frame shift computations
- `/lab/wave-crispr-signal/` - Spectral analysis tools

### Number Theory (/number-theory/)
- `/prime-curve/` - Prime curvature analysis and proof scripts
- `/prime-number-geometry/` - Geometric prime analysis tools

## Validation Scenarios

Always test these core mathematical scenarios after making changes:

### Basic Framework Validation
- Test universal invariance calculation: `from core.axioms import universal_invariance; assert abs(universal_invariance(1.0, 3e8) - 3.33e-09) < 1e-10`
- Test DiscreteZetaShift instantiation: Create instances for n=1 to 100 and verify no exceptions
- Verify high-precision computations work: Check mpmath precision is set to 50 decimal places

### Mathematical Correctness Validation
- Run prime curvature proof: Verify k* = 0.200 and max enhancement = 495.2%
- Test golden ratio transformations: Verify φ ≈ 1.618 calculations
- Validate Mersenne prime generation in proof.py output

### Computational Performance Validation
- Benchmark DiscreteZetaShift: 1000 instances should complete in <3 seconds
- Test visualization generation: hologram.py should complete in <2 seconds
- Verify memory usage remains reasonable for large computations

## Common Tasks and Timing

### Repository Navigation
```bash
# View repository structure
ls -la  # Shows main directories and documentation
find . -name "*.py" | head -20  # Browse Python scripts
cat README.md  # Comprehensive framework documentation (13.9KB)
```

### Documentation Files
- `README.md` - Complete framework description with mathematical foundations
- `MATH.md` - Mathematical detail and theory  
- `PROOFS.md` - Formal mathematical proofs
- `NEXT.md` - Future research directions
- `LICENSE` - MIT license

### Mathematical Computation Patterns
- Core computations use mpmath with 50 decimal precision
- Golden ratio φ = (1 + √5)/2 ≈ 1.618 is central to many calculations
- Optimal curvature parameter k* = 0.200 from current proof analysis
- Prime density enhancement of ~495% is achieved at optimal k*

### Known Dependencies and Limitations
- No traditional build system (pure Python research repository)
- No formal test suite beyond applications/test.py
- Requires PYTHONPATH setup for core module imports
- Some scripts require command-line arguments (use --help to check)
- Visualization scripts use matplotlib with 'Agg' backend for headless environments

## Validation Requirements

### Before Making Changes
- Run basic framework test: `python3 -c "from core.axioms import *; from core.domain import *; print('Core imports successful')"`
- Verify mathematical constants: Check φ, e², and high precision settings

### After Making Changes  
- Run proof validation: `python3 number-theory/prime-curve/proof.py` (should show k* = 0.200)
- Test visualization: `python3 number-theory/prime-curve/hologram.py` (should complete without errors)
- Validate core computations: Test DiscreteZetaShift with known values
- Check computation scaling: Verify 1000 instances complete in reasonable time

### Performance Monitoring
- Monitor memory usage during large computations
- Verify timing remains consistent with baseline measurements
- Check that high-precision arithmetic doesn't cause performance degradation

## Quick Reference

### Common File Locations
- Main proof script: `number-theory/prime-curve/proof.py`
- Core mathematical functions: `core/axioms.py`
- Primary computation class: `core/domain.py`
- Comprehensive test: `applications/test.py`
- Visualization tools: `number-theory/prime-curve/hologram.py`

### Key Mathematical Constants
- Golden ratio φ ≈ 1.618034
- Optimal curvature k* = 0.200 (from proof analysis)
- Max prime enhancement = 495.2% (at optimal k*)
- Speed of light c (used as universal invariant)
- Euler's constant e (e² normalization factor)

### Repository Statistics
- 119 Python files
- 61 documentation files  
- ~109MB repository size
- Mathematical research focus (not software application)