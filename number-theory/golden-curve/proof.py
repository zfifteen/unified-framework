"""
Title: Irreducibility Analysis of Golden-Ratio-Curved Prime Transition Matrices

Author: Big D
Date: 2025-08-01
Description: This executable scientific white paper tests the hypothesis that the transition matrix T(k)
derived from golden-ratio-curved prime values remains irreducible across a range of curvature exponents k.
Irreducibility implies a strongly connected graph, ensuring ergodicity and a unique stationary distribution.
The hypothesis is falsified if T(k) becomes reducible (disconnected components) for any k in the tested range.

Hypothesis: The transition matrix T(k), where theta(p,k) = phi * ((p mod phi) / phi)^k for primes p,
is irreducible for all k in [0.1, 3.0], reflecting invariant connectivity in the prime numberspace.

Note: Falsification occurs if any k yields multiple strongly connected components, indicating frame shifts
disrupt the graph's integrity, akin to relativistic discontinuities at extreme velocities.
"""

import numpy as np
from scipy.sparse.csgraph import connected_components
from sympy import isprime

# Constants
phi = (1 + 5 ** 0.5) / 2
K_RANGE = np.arange(0.1, 3.0, 0.1)

# Curvature Transformation & Matrix Builder
def curvature_transform(n, k):
    """Compute curvature transformation using golden ratio."""
    return phi * ((n % phi) / phi) ** k

def build_transition_matrix(primes, k):
    """Build stochastic transition matrix from curvature states."""
    theta = np.array([curvature_transform(p, k) for p in primes])
    T = np.exp(-np.abs(theta[:, None] - theta[None, :]))
    return T / T.sum(axis=1, keepdims=True)  # Row-normalize

# Irreducibility Test
def test_irreducibility(primes, k_vals):
    """Test if transition matrix is irreducible for all k in k_vals."""
    irreducible_status = []
    for k in k_vals:
        T = build_transition_matrix(primes, k)
        n_components, _ = connected_components(csgraph=T, directed=True, connection='strong')
        irreducible = n_components == 1
        irreducible_status.append((k, irreducible))
    return irreducible_status

# Main Execution Loop
if __name__ == "__main__":
    # Generate primes
    primes = [p for p in range(5, 1000) if isprime(p)]  # Extended range for robustness

    # Run irreducibility test
    results = test_irreducibility(primes, K_RANGE)

    # Falsification check: Fail if any k yields reducible matrix
    is_falsified = any(not status[1] for status in results)
    print("Irreducibility Falsification Test:", "FAIL" if is_falsified else "PASS")

    # Detailed results
    for k, irreducible in results:
        print(f"k = {k:.1f}: Irreducible = {irreducible}")

    # Summary
    if is_falsified:
        print("Hypothesis falsified: Transition matrix becomes reducible at some k, indicating frame-dependent disconnection.")
    else:
        print("Hypothesis passes: Transition matrix remains irreducible across all tested k, supporting invariant connectivity.")