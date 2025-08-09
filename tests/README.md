# Golden Master Test Suite

This test suite ensures reproducibility and detects regressions in the prime-curve density enhancement algorithm.

## Files

### Test Files
- **`test_density_enhancement_minimal.py`**: Main golden master test for density enhancement
- **`run_tests.py`**: Test runner with support for individual and batch testing

### Output Files (Generated)
- **`output/counts_primes_vs_all.csv`**: Bin counts and enhancement percentages
- **`output/bootstrap_midbin_enhancement.csv`**: Bootstrap samples for confidence intervals
- **`output/theta_prime_n1_20_k0.3.csv`**: Sample θ′ transformation values
- **`output/README.md`**: Documentation for output files

## Running Tests

### Run All Tests
```bash
python3 run_tests.py
```

### Run Specific Test
```bash
python3 run_tests.py --test density_enhancement_minimal
```

### Verbose Output
```bash
python3 run_tests.py --verbose
```

### Direct Execution
```bash
python3 test_density_enhancement_minimal.py
```

## Test Parameters

- **N**: 100,000 (number range)
- **k**: 0.3 (curvature parameter)
- **B**: 20 (number of bins)
- **SEED**: 0 (reproducibility seed)
- **φ**: 1.618033988749895 (golden ratio)

## Golden Master Values

These values represent the expected deterministic output:

- **Max Enhancement**: 160.634% ± 0.005
- **Bootstrap CI Lower (2.5%)**: 7.750% ± 0.005  
- **Bootstrap CI Upper (97.5%)**: 681.902% ± 0.005
- **Bootstrap Samples**: 1000 (all finite)

## Algorithm Overview

1. **Generate Data**: Create primes and all integers from 1 to N
2. **Transform**: Apply θ′ = φ * (((n mod φ) / φ) ** k) 
3. **Bin**: Use 20 bins with edges from 0 to φ
4. **Enhance**: Calculate (prime_density - all_density) / all_density * 100
5. **Bootstrap**: Resample primes 1000 times for confidence intervals
6. **Validate**: Check against golden master values

## Reproducibility

The test is designed for exact reproducibility:
- All random operations use SEED=0
- Results are deterministic across multiple runs
- Output files include generation timestamps and parameters
- All floating-point calculations use consistent precision

## Integration

The test can be:
- Run standalone for manual verification
- Imported as a module for programmatic testing
- Executed via the test runner for batch testing
- Integrated into CI/CD pipelines

## Troubleshooting

### Common Issues
- **Import errors**: Ensure numpy, scipy, sympy are installed
- **Timeout**: Test takes ~30-60 seconds for N=100,000
- **Memory**: Requires ~200MB RAM for full dataset
- **Precision**: Results sensitive to numpy version and platform

### Dependencies
```bash
pip install numpy scipy sympy
```

### Expected Runtime
- N=100,000: ~30-60 seconds
- Bootstrap (1000 samples): ~30 seconds
- File I/O: <1 second

## Validation Criteria

✅ **PASS**: All golden master values within tolerance  
❌ **FAIL**: Any value outside expected range  

The test guards against:
- Algorithm changes that affect enhancement calculation
- Bootstrap procedure modifications
- Binning or transformation regressions
- Random seed or reproducibility issues