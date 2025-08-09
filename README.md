# Z Framework: Unified Mathematical Model

The Z Framework is a unified mathematical model bridging physical and discrete domains through the empirical invariance of the speed of light. It leverages the universal form Z = A(B/c) to analyze prime number distributions using geometric constraints and curvature-based geodesics.

## Quick Start

```bash
# Install dependencies
pip3 install numpy pandas matplotlib mpmath sympy scikit-learn statsmodels scipy seaborn plotly

# Set Python path (when working outside repository root)
export PYTHONPATH=/home/runner/work/unified-framework/unified-framework

# Test basic framework
python3 -c "from core.axioms import universal_invariance; print('Test:', universal_invariance(1.0, 3e8))"

# Run prime curvature proof analysis
python3 number-theory/prime-curve/proof.py
```

## Repository Structure

```
unified-framework/
├── core/                    # Core mathematical framework
├── src/                     # Source code and applications
├── docs/                    # 📚 All documentation consolidated here
│   ├── core/               # Main project documentation (README, recent updates)
│   ├── applications/       # Application-specific documentation
│   ├── number-theory/      # Number theory research documentation
│   ├── validation/         # Validation and testing documentation
│   ├── test-finding/       # Test and analysis documentation
│   └── [existing subdirs]  # API, examples, guides, reports, etc.
├── tests/                  # Test suites and validation
└── examples/               # Practical examples and demos
```

## Documentation

**All documentation has been consolidated into the `docs/` directory for better organization:**

- **📖 Main Documentation**: [`docs/core/README.md`](docs/core/README.md) - Complete framework description
- **🔬 Recent Updates**: [`docs/core/RECENT.md`](docs/core/RECENT.md) - Latest research findings
- **⚙️ Configuration**: [`docs/core/copilot-instructions.md`](docs/core/copilot-instructions.md) - Development guidelines
- **🧮 Number Theory**: [`docs/number-theory/`](docs/number-theory/) - Mathematical research documentation
- **🔧 Applications**: [`docs/applications/`](docs/applications/) - Implementation guides
- **🧪 Testing**: [`docs/validation/`](docs/validation/) - Validation documentation

## Key Features

- **Universal Invariance**: Based on speed of light c as fundamental constant
- **Prime Analysis**: 15% density enhancement at optimal curvature k* ≈ 0.3
- **5D Helical Embeddings**: Geometric prime distribution analysis
- **High-Precision Computing**: mpmath with 50 decimal places
- **Spectral Analysis**: Bootstrap confidence bands and form factor computation

## Quick References

### Core Mathematical Operations
```python
from core.axioms import *
from core.domain import DiscreteZetaShift

# Create discrete zeta shift instance
dz = DiscreteZetaShift(10)

# Compute universal invariance
result = universal_invariance(1.0, 3e8)
```

### Key Mathematical Constants
- Golden ratio φ ≈ 1.618034
- Optimal curvature k* ≈ 0.3
- Prime enhancement = 15% (CI [14.6%, 15.4%])

## License

MIT License - See [`LICENSE`](LICENSE) for details.

## Contributing

See [`docs/contributing/`](docs/contributing/) for contribution guidelines.

---

*For complete documentation, mathematical foundations, and research findings, see the [`docs/`](docs/) directory.*