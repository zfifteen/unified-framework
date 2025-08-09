# Z Framework System Instruction Implementation

## Overview

The Z Framework System Instruction (`src/core/system_instruction.py`) establishes the operational logic, empirical rigor, and mathematical principles for unified analysis within the Z Framework. This document describes the implementation and compliance validation.

## Core Principles

### 1. Universal Invariant Formulation: Z = A(B/c)

**Implementation**: `ZFrameworkSystemInstruction.validate_universal_form()`

- **Core Principle**: All observations are normalized to the invariant speed of light `c`
- **Universal Equation**: `Z = A(B/c)` where:
  - `A`: frame-dependent measured quantity
  - `B`: rate or frame shift
  - `c`: universal invariant (speed of light)

**Validation Criteria**:
- c must equal the empirically invariant speed of light (299,792,458 m/s)
- High-precision stability with Δ_n < 10^{-16}
- Proper handling of frame-dependent transformations

### 2. Domain-Specific Forms

#### Physical Domain: Z = T(v/c)
**Implementation**: `ZFrameworkSystemInstruction.validate_physical_domain()`

- **T**: measured time interval (frame-dependent)
- **v**: velocity
- **Empirical basis**: time dilation, Lorentz transformation, experimental results

**Validation Criteria**:
- Causality constraint: |v| < c
- Relativistic effects validation for significant v/c ratios
- Integration with established special relativity predictions

#### Discrete Domain: Z = n(Δ_n/Δ_max)
**Implementation**: `ZFrameworkSystemInstruction.validate_discrete_domain()`

- **n**: frame-dependent integer
- **Δ_n**: measured frame shift κ(n) = d(n) · ln(n+1)/e²
- **Δ_max**: maximum shift (bounded by e² or φ)

**Validation Criteria**:
- Correct curvature formula implementation
- e² normalization for variance minimization
- Proper bounds checking and numerical stability

### 3. Geometric Resolution

**Implementation**: `ZFrameworkSystemInstruction.validate_geometric_resolution()`

- **Geodesic map**: θ'(n, k) = φ · ((n mod φ)/φ)^k
- **Optimal parameter**: k* ≈ 0.3 for ~15% prime density enhancement
- **Golden ratio modular transformation** for replacing fixed ratios

**Validation Criteria**:
- Use of optimal k* ≈ 0.3
- Golden ratio modular arithmetic precision
- Geodesic transformation over fixed ratios
- Result bounds checking: θ'(n,k) ∈ [0, φ)

### 4. Operational Guidance

#### Empirical Validation
**Implementation**: `ZFrameworkSystemInstruction.validate_empirical_claim()`

- Reproducible code and quantitative simulation priority
- Statistical significance requirements (p < 0.05)
- Confidence intervals for all claims
- Clear distinction between hypotheses and validated results

#### Scientific Communication
**Implementation**: `ZFrameworkSystemInstruction.validate_scientific_communication()`

- Precise scientific tone maintenance
- Proper evidence citations for claims
- Detection of unsupported assertions
- Hypothesis labeling requirements

## Integration with Core Modules

### Axioms Module (`src/core/axioms.py`)

Key functions enhanced with system instruction compliance:
- `universal_invariance()`: Universal Z form computation
- `theta_prime()`: Geometric resolution transformation
- `UniversalZForm` class: High-precision Z form implementation
- `PhysicalDomainZ` class: Physical domain specialization

### Domain Module (`src/core/domain.py`)

- `DiscreteZetaShift` class: Discrete domain implementation
- 5D helical embeddings with system instruction compliance
- Curvature-based geodesic parameter computation

## Compliance Enforcement

### Automatic Validation

The `@enforce_system_instruction` decorator provides automatic compliance checking:

```python
@enforce_system_instruction
def my_z_framework_function(args):
    # Function implementation
    return result
```

### Manual Validation

For comprehensive analysis:

```python
from src.core.system_instruction import get_system_instruction

system_instruction = get_system_instruction()
compliance = system_instruction.verify_full_compliance(operation_data)
```

## Testing and Validation

### Test Suite: `tests/test_system_instruction.py`

Comprehensive test coverage includes:
- System constants validation
- Universal invariant formulation testing
- Domain-specific form compliance
- Geometric resolution verification
- Empirical claim validation
- Scientific communication standards
- Full compliance verification

### Running Tests

```bash
cd /home/runner/work/unified-framework/unified-framework
PYTHONPATH=/home/runner/work/unified-framework/unified-framework python3 tests/test_system_instruction.py
```

Expected output: All 9 tests should pass with "🎉 All Z Framework system instruction tests passed!"

## Constants and Thresholds

### System Constants
- **c**: 299,792,458.0 (speed of light)
- **e²**: exp(2) ≈ 7.389 (discrete normalization)
- **φ**: (1 + √5)/2 ≈ 1.618 (golden ratio)
- **k***: 0.3 (optimal curvature parameter)

### Validation Thresholds
- **Precision**: 10^{-16} (high-precision requirement)
- **Enhancement**: 15% (target prime density enhancement)
- **Variance**: 0.118 (target geodesic variance)
- **Confidence**: 95% (default confidence level for claims)

## Usage Examples

### Basic Universal Form Validation
```python
from src.core.system_instruction import get_system_instruction

system_instruction = get_system_instruction()
validation = system_instruction.validate_universal_form(
    A=lambda x: 2.0 * x,  # Linear transformation
    B=1.5e8,              # Rate quantity
    c=299792458.0         # Speed of light
)
assert validation['universal_form_compliant']
```

### Empirical Claim Validation
```python
claim = "15% prime density enhancement at k* ≈ 0.3"
evidence = {
    'statistical_measure': 'enhancement_percentage',
    'confidence_interval': [14.6, 15.4],
    'p_value': 1e-6,
    'sample_size': 1000,
    'reproducible_code': 'src/number-theory/prime-curve/proof.py'
}

validation = system_instruction.validate_empirical_claim(claim, evidence)
assert validation['empirically_substantiated']
```

## Compliance Scoring

The system instruction provides numerical compliance scoring:
- **Score ≥ 0.8**: Fully compliant
- **Score 0.5-0.8**: Partially compliant with warnings
- **Score < 0.5**: Non-compliant with critical violations

## Integration Benefits

1. **Systematic Validation**: Automatic checking of Z Framework principles
2. **Empirical Rigor**: Enforcement of evidence-based claims
3. **Cross-Domain Consistency**: Unified validation across physical and discrete domains
4. **Scientific Standards**: Automated communication quality checking
5. **Precision Assurance**: High-precision numerical stability validation

## Future Enhancements

The system instruction framework is designed to be extensible:
- Additional domain-specific validations
- Enhanced statistical testing requirements
- Integration with automated theorem proving
- Extended precision validation for extreme computations
- Machine learning-based claim validation