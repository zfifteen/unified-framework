#!/usr/bin/env python3
"""
Test script to validate k* = 3.33 produces ~15% enhancement
Based on NUMERICAL_STABILITY_VALIDATION_REPORT.md methodology
"""

import numpy as np
from sympy import sieve

def frame_shift(n_vals, k):
    """Apply golden ratio frame shift transformation"""
    phi = (1 + np.sqrt(5)) / 2
    mod_phi = np.mod(n_vals, phi) / phi
    return phi * np.power(mod_phi, k)

def compute_enhancement(k_value, N=10000):
    """Compute density enhancement for given k value"""
    # Generate test data
    integers = np.arange(1, N + 1)
    primes = np.array(list(sieve.primerange(2, N + 1)))
    
    # Apply transformation
    phi = (1 + np.sqrt(5)) / 2
    theta_all = frame_shift(integers, k_value)
    theta_primes = frame_shift(primes, k_value)
    
    # Compute enhancement using 20 bins
    bins = np.linspace(0, phi, 20 + 1)
    all_counts, _ = np.histogram(theta_all, bins=bins)
    prime_counts, _ = np.histogram(theta_primes, bins=bins)
    
    all_density = all_counts / len(theta_all)
    prime_density = prime_counts / len(theta_primes)
    
    # Compute enhancements, handling division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        enhancement = (prime_density - all_density) / all_density * 100
    
    # Return max finite enhancement
    finite_enhancements = enhancement[np.isfinite(enhancement)]
    if len(finite_enhancements) == 0:
        return 0.0
    return np.max(finite_enhancements)

def main():
    print("Testing curvature exponent k* = 3.33 validation")
    print("=" * 50)
    
    # Test k = 3.33 (our expected optimal)
    k_test = 3.33
    enhancement_333 = compute_enhancement(k_test)
    print(f"Enhancement at k = {k_test:.2f}: {enhancement_333:.1f}%")
    
    # Test k = 0.3 (old value)
    k_old = 0.3
    enhancement_03 = compute_enhancement(k_old)
    print(f"Enhancement at k = {k_old:.1f}: {enhancement_03:.1f}%")
    
    # Test the reciprocal relationship
    k_reciprocal = 1.0 / 3.33
    enhancement_recip = compute_enhancement(k_reciprocal)
    print(f"Enhancement at k = 1/3.33 = {k_reciprocal:.3f}: {enhancement_recip:.1f}%")
    
    print("\nValidation Results:")
    print(f"- k* = 3.33 gives {enhancement_333:.1f}% enhancement")
    print(f"- k = 0.3 gives {enhancement_03:.1f}% enhancement") 
    print(f"- k = 1/3.33 gives {enhancement_recip:.1f}% enhancement")
    
    # Check if 3.33 is closer to expected 15%
    target = 15.0
    diff_333 = abs(enhancement_333 - target)
    diff_03 = abs(enhancement_03 - target)
    diff_recip = abs(enhancement_recip - target)
    
    print(f"\nDistance from target 15%:")
    print(f"- k = 3.33: {diff_333:.1f}% difference")
    print(f"- k = 0.3: {diff_03:.1f}% difference")
    print(f"- k = 1/3.33: {diff_recip:.1f}% difference")
    
    closest_k = min([(diff_333, "3.33"), (diff_03, "0.3"), (diff_recip, "1/3.33")])
    print(f"\nClosest to 15% target: k = {closest_k[1]} (difference: {closest_k[0]:.1f}%)")

if __name__ == "__main__":
    main()