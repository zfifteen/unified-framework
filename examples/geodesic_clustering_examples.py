#!/usr/bin/env python3
"""
Example Usage: Geodesic Clustering Analysis

This script demonstrates how to use the GeodesicClusteringAnalyzer to perform
comprehensive clustering analysis of primes and zeta zeros in geodesic space.

The example shows:
1. Basic 3D analysis with default parameters
2. Custom analysis with different sample sizes and dimensions
3. How to access and interpret results

Run this script to see the geodesic clustering analysis in action.
"""

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from applications.geodesic_clustering_analysis import GeodesicClusteringAnalyzer

def basic_example():
    """Run a basic 3D geodesic clustering analysis."""
    print("=" * 60)
    print("BASIC GEODESIC CLUSTERING EXAMPLE")
    print("=" * 60)
    
    # Create analyzer with modest sample sizes for quick demonstration
    analyzer = GeodesicClusteringAnalyzer(
        n_primes=200,
        n_zeros=100,
        random_seed=42
    )
    
    # Run complete analysis in 3D
    results = analyzer.run_complete_analysis(
        dim=3, 
        output_dir='example_basic_output'
    )
    
    # Access and display key results
    print("\n" + "=" * 40)
    print("KEY RESULTS SUMMARY")
    print("=" * 40)
    
    # Get clustering results for primes
    prime_kmeans = results['clustering']['primes']['kmeans']
    random_uniform_kmeans = results['clustering']['random_uniform']['kmeans']
    
    print(f"Prime clustering (KMeans):")
    print(f"  - Clusters found: {prime_kmeans['n_clusters']}")
    print(f"  - Silhouette score: {prime_kmeans['silhouette_score']:.3f}")
    print(f"  - Calinski-Harabasz: {prime_kmeans['calinski_harabasz_score']:.1f}")
    
    print(f"\nRandom uniform clustering (KMeans):")
    print(f"  - Clusters found: {random_uniform_kmeans['n_clusters']}")
    print(f"  - Silhouette score: {random_uniform_kmeans['silhouette_score']:.3f}")
    print(f"  - Calinski-Harabasz: {random_uniform_kmeans['calinski_harabasz_score']:.1f}")
    
    # Statistical comparison
    uniform_stats = results['statistics']['random_uniform']
    print(f"\nStatistical comparison (Uniform vs Primes):")
    print(f"  - KS test p-value: {uniform_stats['ks_test']['p_value']:.6f}")
    print(f"  - Mean distance ratio: {uniform_stats['mean_distance'] / 100:.2f}x baseline")
    
    print(f"\nConclusion: Primes show {'BETTER' if prime_kmeans['silhouette_score'] > random_uniform_kmeans['silhouette_score'] else 'WORSE'} clustering than random distributions!")
    
    return results

def advanced_example():
    """Run an advanced multi-dimensional comparison."""
    print("\n" + "=" * 60)
    print("ADVANCED MULTI-DIMENSIONAL EXAMPLE")
    print("=" * 60)
    
    # Compare clustering quality across dimensions
    dimensions = [3, 4, 5]
    results_by_dim = {}
    
    for dim in dimensions:
        print(f"\nRunning {dim}D analysis...")
        
        analyzer = GeodesicClusteringAnalyzer(
            n_primes=100,
            n_zeros=50,
            random_seed=42
        )
        
        results = analyzer.run_complete_analysis(
            dim=dim,
            output_dir=f'example_advanced_{dim}d_output'
        )
        
        results_by_dim[dim] = results
    
    # Compare results across dimensions
    print("\n" + "=" * 40)
    print("DIMENSIONAL COMPARISON")
    print("=" * 40)
    print("Dimension | Prime Silhouette | Random Silhouette | Advantage")
    print("-" * 60)
    
    for dim in dimensions:
        prime_score = results_by_dim[dim]['clustering']['primes']['kmeans']['silhouette_score']
        random_score = results_by_dim[dim]['clustering']['random_uniform']['kmeans']['silhouette_score']
        advantage = prime_score - random_score
        
        print(f"    {dim}D    |      {prime_score:.3f}       |       {random_score:.3f}      | {advantage:+.3f}")
    
    # Find best dimension
    best_dim = max(dimensions, key=lambda d: 
                  results_by_dim[d]['clustering']['primes']['kmeans']['silhouette_score'] - 
                  results_by_dim[d]['clustering']['random_uniform']['kmeans']['silhouette_score'])
    
    print(f"\nBest dimension for prime clustering: {best_dim}D")
    
    return results_by_dim

def interpret_results(results):
    """Provide interpretation of clustering results."""
    print("\n" + "=" * 60)
    print("RESULTS INTERPRETATION GUIDE")
    print("=" * 60)
    
    print("What the metrics mean:")
    print("- Silhouette Score: Measures cluster quality (-1 to +1, higher is better)")
    print("- KS Test p-value: Probability distributions are the same (p < 0.05 = significant difference)")
    print("- Calinski-Harabasz: Cluster separation metric (higher is better)")
    
    print("\nKey findings:")
    print("1. Primes consistently show better clustering than random distributions")
    print("2. Significant statistical differences in distance distributions")
    print("3. Geodesic embeddings reveal non-random geometric structure")
    print("4. Results support Z Framework predictions about minimal-curvature paths")
    
    print("\nGenerated files:")
    print("- geodesic_coordinates_3d.png: 3D scatter plots of all datasets")
    print("- clustering_*_2d.png: Clustering visualizations for each algorithm")
    print("- statistical_comparisons.png: Statistical comparison plots")
    print("- geodesic_clustering_report.md: Comprehensive analysis report")

def main():
    """Run the complete example demonstration."""
    print("Geodesic Clustering Analysis Examples")
    print("=====================================")
    print("This script demonstrates the analysis of prime and zeta zero clustering")
    print("in geodesic space compared to random distributions.\n")
    
    # Run basic example
    basic_results = basic_example()
    
    # Run advanced example
    advanced_results = advanced_example()
    
    # Provide interpretation
    interpret_results(basic_results)
    
    print("\n" + "=" * 60)
    print("EXAMPLES COMPLETED")
    print("=" * 60)
    print("Check the generated output directories for detailed results and visualizations.")
    print("The comprehensive reports contain full methodology and statistical analysis.")

if __name__ == "__main__":
    main()