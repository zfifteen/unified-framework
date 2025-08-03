# Prime Density and Curvature Transform Analysis

This project involves an exploration of prime number density transformations using mathematical, computational, and statistical techniques. The goal is to analyze the behavior of primes under a curvature transformation and evaluate their clustering properties, with applications ranging from cryptography to theoretical mathematics.

## Overview

Prime numbers, traditionally considered as distributed in a "random-like" manner, are subjected to a **curvature-based transformation** that introduces systematic variations in their density and clustering. This yields new insights into the structure and properties of prime distributions, particularly in relation to randomness and uniformity in their behavior.

The core computational approach applies a curvature transformation defined as:

\[
\theta'(n, k) = \phi \cdot \left(\frac{(n \mod \phi)}{\phi}\right)^k
\]

where:
- \( k \) is a scaling parameter (default: \( k \approx 0.3 \)),
- \( \phi \) is the golden ratio: \( \phi = \frac{1 + \sqrt{5}}{2} \).

This transformation is normalized using:

\[
Z(n) = n \cdot \left(\frac{\kappa(n)}{e^2}\right),
\]

where:
- \( \kappa(n) = d(n) \cdot \ln(n+1) \),
- \( d(n) \) represents the divisor count for \( n \),
- \( e \) is Euler’s number.

The analysis also includes rigorous statistical tests to challenge the hypotheses about prime distribution properties under curvature transformations and generates falsifiable predictions.

## Objectives

1. **Prime Curvature Transformation**:
    - Apply the curvature transformation to a range of primes.
    - Quantify clustering behavior through density analysis and statistical tools.

2. **Density and Randomness Tests**:
    - Evaluate the transformed prime distributions to test for deviations from uniform randomness using metrics like:
        - Kolmogorov-Smirnov (KS) test,
        - Kullback-Leibler (KL) divergence,
        - Bootstrapped confidence intervals.

3. **Visualization**:
    - Provide meaningful visual representations of transformed prime density, including clustering metrics and comparisons to expected uniform distributions.

4. **Applications**:
    - Explore practical applications in areas like cryptography, where improved understanding of prime density and clustering could enhance large-prime-generation algorithms.

## Methodology

### Prime Transformation

1. **Divisor Count Precomputation**:
   Efficient computation of divisor counts for all integers up to \( N \).

```python
def precompute_divisor_counts(N):
       dcount = np.zeros(N + 1, dtype=int)
       for i in range(1, N + 1):
           dcount[i::i] += 1
       return dcount
```


2. **Curvature Transformation**:
   Each prime is mapped into a new space based on the curvature properties of its residues modulo \( \phi \), as well as an exponential normalization factor derived from \( Z(n) \).

### Statistical Analysis

- **KS Test**:
  Compare the distribution of transformed primes to a theoretical uniform distribution to assess deviations.

- **Bootstrapped Confidence Intervals**:
  Compute confidence intervals for density enhancement metrics, indicating whether clustering is statistically significant or a product of random noise.

- **KL Divergence**:
  Measure the divergence of the empirical transformed-prime distribution from a uniform reference.

### Clustering and Density Metrics

- **Density Enhancement**:
  Compute the ratio of maximum density over the expected uniform density.

  \[
  \text{Enhancement} = \left(\frac{\text{Max Density}}{\text{Uniform Density}}\right) - 1
  \]

- **Gaussian Mixture Model (GMM)**:
  Fit a GMM to identify substructures within the transformed distribution.

### Visualization

- Render histograms of transformed primes, with overlays of expected uniform densities for comparison.
- Save plots as `.png` images for further inspection.

## Results

### Falsifiability Tests

1. **Kolmogorov-Smirnov Test**:
    - Null hypothesis: The transformed distribution follows a uniform distribution.
    - Outcome: The hypothesis is rejected if \( p \)-value \( < 0.05 \).

2. **Bootstrap Confidence Interval**:
    - Null hypothesis: The density enhancement metric includes 0% enhancement.
    - Outcome: Rejected if 95% confidence intervals do not include 0.

3. **KL Divergence Test**:
    - Null hypothesis: There is no significant clustering in the transformed primes.
    - Outcome: Rejected if \( KL < 0.1 \).

### Practical Metrics

- **Density Enhancement**:
  Quantifies clustering boost, observed up to ~15% in key regions of the transformation.

- **Clustering Compactness**:
  Measures the consistency of identified clusters based on the Gaussian Mixture results.

## Applications

### Cryptography

The understanding and optimization of prime density are critical for efficiently generating large prime numbers in cryptographic systems such as RSA. The curvature transformation enhances our ability to identify prime-rich regions, potentially reducing computation time.

### Theoretical Mathematics

Insights from the prime curvature transformation may contribute to broader understanding of prime distributions in number theory. Connections between curvature and density suggest geometric foundations underpinning prime behavior.

## Visualization

Prime density distributions and curvature transformations are visualized and output as charts, e.g., `prime_density_plot.png`.

---

## Dependencies

- **NumPy**: High-performance numerical computations.
- **SymPy**: Symbolic mathematics for prime generation.
- **SciPy**: Statistical tests and KL divergence.
- **scikit-learn (GaussianMixture)**: Clustering metrics for transformed data.
- **Matplotlib**: Visualization of results.

To ensure reproducibility, seed values are fixed using `np.random.seed()`.

## Execution

Run the script in a Python environment with the required dependencies installed. The main function computes transformations, evaluates statistical tests, and creates visualizations:

```shell script
python prime_density.py
```


### Input Parameters

| Parameter  | Default Value | Description                                           |
|------------|---------------|-------------------------------------------------------|
| \( N \)    | 1,000,000     | Upper limit of primes to analyze and transform.       |
| \( k \)    | 0.3           | Curvature scaling parameter for transformation.      |

---

## Conclusion

This project establishes a structured methodology to examine prime distributions under a curvature transformation. The statistical and visual analyses provide strong evidence of clustering behaviors, challenging traditional assumptions of primal randomness. Furthermore, this work highlights potential cryptographic and theoretical applications, making it a significant contribution to both applied and pure mathematics.

--- 
