"""
domain.py: Implementation of the Zeta Shift inheritance model based on the Universal Form Transformer.

This module defines classes that faithfully reproduce the inheritance model for the Z definition,
unifying relativistic physics with discrete mathematics by normalizing observations against invariant limits.

Key Aspects:
- Universal Form: Z = A(B/C), where A is the reference frame-dependent measured quantity,
  B is the rate, and C is the universal invariant (speed of light or analogous limit).
- Physical Domain: Z = T(v/c), specializing A=T (measured quantity), B=v (velocity), C=c (speed of light).
- Discrete Domain: Z = n(Δₙ/Δmax), specializing A=n (integer observation), B=Δₙ (frame shift at n), C=Δmax (max frame shift).

The model reveals shared geometric topology across domains, where curvature arises from frame-dependent shifts.
In geometry, we replace hard-coded ratios with natural invariants like the golden ratio or e, but here we adhere to c as the core invariant.
"""

from abc import ABC
import math

class UniversalZetaShift(ABC):
    """
    Abstract base class for the Universal Form Z = A(B/C).

    A = reference frame–dependent measured quantity
    B = Rate
    C = The speed of light is the only universal invariant that limits all regimes. All other regime-invariants,
        large or small, are relative to c, the universal limit of B.

    This class provides the core computation, with subclasses specializing for domains.
    """

    def __init__(self, a, b, c):
        if c == 0:
            raise ValueError("Universal invariant C cannot be zero.")
        self.a = a  # Reference frame-dependent measured quantity
        self.b = b  # Rate
        self.c = c  # Universal invariant

    def compute_z(self):
        """
        Compute Z according to the universal form Z = A(B/C).
        """
        return self.a * (self.b / self.c)

class PhysicalZetaShift(UniversalZetaShift):
    """
    Specialization for the Physical Domain: Z = T(v/c)

    T = reference frame–dependent measured quantity (maps to A)
    v = velocity (maps to B)
    c = speed of light (fixed universal invariant)

    Empirically, this implies time T as a frame-biased measure scaled by velocity relative to light's invariance.
    """

    # SPEED_OF_LIGHT = 299792458.0  # m/s, the universal invariant c
    SPEED_OF_LIGHT = 1/137.035999206

    def __init__(self, t, v):
        super().__init__(t, v, self.SPEED_OF_LIGHT)

class DiscreteZetaShift(UniversalZetaShift):
    """
    Specialization for the Discrete Domain: Z = n(Δₙ/Δmax)

    n = reference frame–dependent integer observation (maps to A)
    Δₙ = measured frame shift at position n (maps to B)
    Δmax = maximum possible frame shift in the domain (maps to C, analogous to c)

    Empirically, this computes a 'relativistic density' for discrete sequences like primes, highlighting clustering
    patterns in a curved numerical landscape.
    """

    def __init__(self, n, delta_n, delta_max):
        super().__init__(n, delta_n, delta_max)

# Example usage (commented out; for demonstration)
if __name__ == "__main__":
    # Physical example
    physical = PhysicalZetaShift(t=1.0, v=150000000.0)  # Example: T=1s, v=0.5c approx
    print(f"Physical Z: {physical.compute_z()}")

    # Discrete example (e.g., for primes, delta_n could be gap, delta_max theoretical max gap)
    discrete = DiscreteZetaShift(n=5, delta_n=2, delta_max=10)
    print(f"Discrete Z: {discrete.compute_z()}")