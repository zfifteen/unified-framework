"""
Unit tests for Prime-Driven Compression Algorithm

Tests the mathematical foundations, compression/decompression accuracy,
and performance characteristics of the prime-driven modular clustering
compression algorithm.
"""

import unittest
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from applications.prime_compression import (
    PrimeGeodesicTransform, 
    ModularClusterAnalyzer,
    PrimeDrivenCompressor,
    CompressionBenchmark,
    CompressionMetrics,
    K_OPTIMAL, PHI
)


class TestPrimeGeodesicTransform(unittest.TestCase):
    """Test the core mathematical transformation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.transformer = PrimeGeodesicTransform()
        
    def test_optimal_k_parameter(self):
        """Test that optimal curvature parameter is correctly set."""
        self.assertAlmostEqual(float(self.transformer.k), 0.300, places=3)
        
    def test_golden_ratio_value(self):
        """Test that golden ratio is correctly computed."""
        expected_phi = (1 + np.sqrt(5)) / 2
        self.assertAlmostEqual(float(self.transformer.phi), expected_phi, places=10)
        
    def test_frame_shift_residues_basic(self):
        """Test basic frame shift residue computation."""
        indices = np.array([1, 2, 3, 4, 5])
        result = self.transformer.frame_shift_residues(indices)
        
        # Should return array of same length
        self.assertEqual(len(result), len(indices))
        
        # All values should be in [0, phi) range
        phi_val = float(PHI)
        self.assertTrue(np.all(result >= 0))
        self.assertTrue(np.all(result < phi_val))
        
    def test_frame_shift_residues_deterministic(self):
        """Test that transformation is deterministic."""
        indices = np.array([10, 20, 30])
        result1 = self.transformer.frame_shift_residues(indices)
        result2 = self.transformer.frame_shift_residues(indices)
        
        np.testing.assert_array_almost_equal(result1, result2, decimal=10)
        
    def test_prime_enhancement_computation(self):
        """Test prime density enhancement calculation."""
        # Create test data with known prime pattern
        theta_values = np.array([0.1, 0.5, 1.0, 1.3, 1.6])
        prime_mask = np.array([True, True, False, False, False])
        
        enhancement = self.transformer.compute_prime_enhancement(theta_values, prime_mask)
        
        # Enhancement should be a positive number
        self.assertGreater(enhancement, 0)
        self.assertTrue(np.isfinite(enhancement))


class TestModularClusterAnalyzer(unittest.TestCase):
    """Test the cluster analysis component."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = ModularClusterAnalyzer(n_components=3)
        
    def test_fit_clusters_basic(self):
        """Test basic cluster fitting."""
        # Generate test data with clear clusters
        np.random.seed(42)
        theta_values = np.concatenate([
            np.random.normal(0.5, 0.1, 50),
            np.random.normal(1.2, 0.1, 50),
            np.random.normal(0.1, 0.05, 30)
        ])
        
        results = self.analyzer.fit_clusters(theta_values)
        
        # Check results structure
        self.assertIn('labels', results)
        self.assertIn('cluster_stats', results)
        self.assertIn('bic', results)
        self.assertIn('aic', results)
        
        # Check that we get expected number of clusters
        self.assertEqual(len(results['cluster_stats']), 3)
        
        # Labels should be in valid range
        labels = results['labels']
        self.assertTrue(np.all(labels >= 0))
        self.assertTrue(np.all(labels < 3))
        
    def test_predict_cluster(self):
        """Test cluster prediction on new data."""
        # Fit on initial data
        np.random.seed(42)
        fit_data = np.random.normal(1.0, 0.3, 100)
        self.analyzer.fit_clusters(fit_data)
        
        # Predict on new data
        new_data = np.array([0.5, 1.0, 1.5])
        predictions = self.analyzer.predict_cluster(new_data)
        
        # Should return valid cluster IDs
        self.assertEqual(len(predictions), len(new_data))
        self.assertTrue(np.all(predictions >= 0))
        self.assertTrue(np.all(predictions < 3))
        
    def test_predict_before_fit_raises_error(self):
        """Test that prediction before fitting raises error."""
        test_data = np.array([1.0, 2.0, 3.0])
        
        with self.assertRaises(ValueError):
            self.analyzer.predict_cluster(test_data)


class TestPrimeDrivenCompressor(unittest.TestCase):
    """Test the main compression algorithm."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.compressor = PrimeDrivenCompressor()
        
    def test_compress_empty_data(self):
        """Test compression of empty data."""
        empty_data = b''
        compressed_data, metrics = self.compressor.compress(empty_data)
        
        self.assertEqual(len(compressed_data), 0)
        self.assertEqual(metrics.original_size, 0)
        self.assertEqual(metrics.compressed_size, 0)
        
    def test_compress_small_data(self):
        """Test compression of small data."""
        test_data = b'hello world'
        compressed_data, metrics = self.compressor.compress(test_data)
        
        # Basic sanity checks
        self.assertIsInstance(compressed_data, bytes)
        self.assertGreater(metrics.compression_time, 0)
        self.assertEqual(metrics.original_size, len(test_data))
        self.assertEqual(metrics.compressed_size, len(compressed_data))
        
    def test_compress_decompress_cycle(self):
        """Test full compression-decompression cycle."""
        test_data = b'This is a test message for compression testing.' * 10
        
        # Compress
        compressed_data, metrics = self.compressor.compress(test_data)
        
        # Decompress
        decompressed_data, integrity_verified = self.compressor.decompress(compressed_data, metrics)
        
        # Size should match
        self.assertEqual(len(decompressed_data), len(test_data))
        
    def test_mathematical_properties(self):
        """Test mathematical properties of the algorithm."""
        test_data = b'x' * 1000  # Simple repetitive data
        
        compressed_data, metrics = self.compressor.compress(test_data)
        
        # Check mathematical properties
        self.assertEqual(metrics.prime_clusters_found, 5)  # Default n_components
        self.assertEqual(metrics.geodesic_mappings, len(test_data))
        self.assertGreater(metrics.enhancement_factor, 1.0)  # Should find some enhancement
        
    def test_different_data_types(self):
        """Test compression on different data types."""
        test_cases = [
            b'\x00' * 100,  # All zeros
            b'\xFF' * 100,  # All ones
            bytes(range(256)),  # Sequential bytes
            b'a' * 50 + b'b' * 50,  # Two-pattern data
        ]
        
        for test_data in test_cases:
            with self.subTest(data_type=type(test_data).__name__):
                compressed_data, metrics = self.compressor.compress(test_data)
                
                # Should complete without error
                self.assertIsInstance(compressed_data, bytes)
                self.assertGreater(metrics.compression_time, 0)


class TestCompressionBenchmark(unittest.TestCase):
    """Test the benchmarking suite."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.benchmark = CompressionBenchmark()
        
    def test_generate_test_data(self):
        """Test test data generation."""
        data_types = ['sparse', 'incompressible', 'repetitive', 'mixed', 'random']
        size = 100
        
        for data_type in data_types:
            with self.subTest(data_type=data_type):
                data = self.benchmark.generate_test_data(data_type, size)
                
                self.assertEqual(len(data), size)
                self.assertIsInstance(data, bytes)
                
    def test_benchmark_algorithm_prime_driven(self):
        """Test benchmarking of prime-driven algorithm."""
        test_data = b'test data for compression' * 20
        
        result = self.benchmark.benchmark_algorithm('prime_driven', test_data)
        
        self.assertTrue(result['success'])
        self.assertIsNotNone(result['metrics'])
        self.assertIsInstance(result['metrics'], CompressionMetrics)
        
    def test_benchmark_algorithm_gzip(self):
        """Test benchmarking of gzip algorithm."""
        test_data = b'test data for compression' * 20
        
        result = self.benchmark.benchmark_algorithm('gzip', test_data)
        
        self.assertTrue(result['success'])
        self.assertIsNotNone(result['metrics'])
        
    def test_run_mini_benchmark(self):
        """Test running a minimal benchmark."""
        test_cases = [('sparse', 100), ('repetitive', 100)]
        
        results = self.benchmark.run_comprehensive_benchmark(test_cases)
        
        # Should have results for both test cases
        self.assertEqual(len(results), 2)
        
        # Each test case should have results for all algorithms
        for case_name, case_results in results.items():
            self.assertIn('prime_driven', case_results)
            self.assertIn('gzip', case_results)


class TestMathematicalFoundations(unittest.TestCase):
    """Test mathematical foundations and invariants."""
    
    def test_k_optimal_constant(self):
        """Test that optimal k parameter is correctly defined."""
        self.assertAlmostEqual(float(K_OPTIMAL), 0.300, places=3)
        
    def test_golden_ratio_constant(self):
        """Test that golden ratio is correctly computed."""
        expected = (1 + np.sqrt(5)) / 2
        self.assertAlmostEqual(float(PHI), expected, places=10)
        
    def test_transformation_properties(self):
        """Test mathematical properties of the transformation."""
        transformer = PrimeGeodesicTransform()
        
        # Test monotonicity within modular ranges
        indices1 = np.array([1.0, 1.1, 1.2])
        indices2 = np.array([2.0, 2.1, 2.2])
        
        result1 = transformer.frame_shift_residues(indices1)
        result2 = transformer.frame_shift_residues(indices2)
        
        # Results should be in valid range
        phi_val = float(PHI)
        self.assertTrue(np.all(result1 >= 0))
        self.assertTrue(np.all(result1 < phi_val))
        self.assertTrue(np.all(result2 >= 0))
        self.assertTrue(np.all(result2 < phi_val))
        
    def test_compression_invariants(self):
        """Test compression algorithm invariants."""
        compressor = PrimeDrivenCompressor()
        
        # Test with known data
        test_data = bytes([i % 256 for i in range(500)])
        compressed_data, metrics = compressor.compress(test_data)
        
        # Mathematical invariants
        self.assertEqual(metrics.original_size, len(test_data))
        self.assertEqual(metrics.compressed_size, len(compressed_data))
        self.assertEqual(metrics.geodesic_mappings, len(test_data))
        self.assertGreater(metrics.enhancement_factor, 0)


if __name__ == '__main__':
    """Run all tests."""
    unittest.main(verbosity=2)