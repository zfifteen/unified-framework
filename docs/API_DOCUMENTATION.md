# Universal Invariant Scoring Engine API Documentation

The Universal Invariant Scoring Engine provides Z-invariant based scoring and density analysis for arbitrary sequences including numerical, biological, and network data.

## Table of Contents
- [Overview](#overview)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Python Client](#python-client)
- [Data Types](#data-types)
- [Examples](#examples)
- [Mathematical Background](#mathematical-background)

## Overview

The Z-score API leverages the Z Framework's mathematical foundations to provide:

- **Universal Scoring**: Z-invariant scores comparable across different data domains
- **Density Analysis**: Prime-based algorithms with 15% enhancement at optimal curvature k*≈0.3
- **Anomaly Detection**: Geometric and statistical anomaly detection using curvature-based methods
- **Cross-Domain Normalization**: Enables benchmarking between numerical, biological, and network data
- **High-Precision Computing**: Uses mpmath with 50 decimal places for numerical stability

## Quick Start

### 1. Start the API Server

```bash
cd /path/to/unified-framework
export PYTHONPATH=$(pwd)
python3 -m src.api.server
```

The server will start on `http://localhost:5000` by default.

### 2. Test with curl

```bash
# Health check
curl http://localhost:5000/health

# Score a numerical sequence
curl -X POST http://localhost:5000/api/score \
  -H "Content-Type: application/json" \
  -d '{"data": [1, 2, 3, 4, 5], "data_type": "numerical"}'

# Score a biological sequence
curl -X POST http://localhost:5000/api/score \
  -H "Content-Type: application/json" \
  -d '{"data": "ATGCATGC", "data_type": "biological"}'
```

### 3. Use Python Client

```python
from src.api.client import ZScoreAPIClient

# Initialize client
client = ZScoreAPIClient("http://localhost:5000")

# Score different data types
numerical_score = client.score_numerical([1, 2, 3, 4, 5])
biological_score = client.score_biological("ATGCATGC")
network_score = client.score_network([[0, 1], [1, 0]])

print(f"Scores: {numerical_score:.3f}, {biological_score:.3f}, {network_score:.3f}")
```

## API Reference

### Base URL
`http://localhost:5000`

### Endpoints

#### GET /health
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "service": "Universal Invariant Scoring Engine",
  "version": "1.0.0"
}
```

#### POST /api/score
Score a single sequence.

**Request Body:**
```json
{
  "data": [1, 2, 3, 4, 5],
  "data_type": "numerical"  // optional: "numerical", "biological", "network", "auto"
}
```

**Response:**
```json
{
  "success": true,
  "result": {
    "metadata": {
      "type": "numerical",
      "length": 5,
      "mean": 3.0,
      "std": 1.4142135623730951,
      "min": 1.0,
      "max": 5.0
    },
    "z_invariant_score": 54.745542,
    "z_scores": {
      "mean_z": 109.491084,
      "std_z": 0.0,
      "max_z": 109.491084,
      "min_z": 109.491084,
      "universal_invariance": 0.0,
      "composite_score": 54.745542
    },
    "density_metrics": {
      "basic_density": 1.25,
      "cluster_variance": 0.0625,
      "prime_density": 0.0,
      "enhancement_factor": 1.0
    },
    "anomaly_scores": {
      "statistical_anomalies": 0.0,
      "curvature_anomalies": 1.0,
      "frame_anomalies": 0.0,
      "composite_anomaly_score": 0.3333333333333333
    },
    "normalized_sequence": [0.0, 0.25, 0.5, 0.75, 1.0],
    "sequence_length": 5
  }
}
```

#### POST /api/batch_score
Score multiple sequences in batch.

**Request Body:**
```json
{
  "data_list": [
    [1, 2, 3, 4, 5],
    "ATGCATGC",
    [[0, 1], [1, 0]]
  ],
  "data_types": ["numerical", "biological", "network"]  // optional
}
```

**Response:**
```json
{
  "success": true,
  "results": [
    {
      "batch_index": 0,
      "metadata": {...},
      "z_invariant_score": 54.745542,
      ...
    },
    ...
  ],
  "total_processed": 3
}
```

#### POST /api/analyze
Comprehensive sequence analysis with detailed breakdown.

**Request Body:**
```json
{
  "data": [1, 2, 3, 4, 5],
  "data_type": "numerical",
  "include_normalized": true,
  "detailed_metrics": true
}
```

**Response:**
```json
{
  "success": true,
  "analysis": {
    "metadata": {...},
    "summary": {
      "z_invariant_score": 54.745542,
      "composite_anomaly_score": 0.333,
      "enhancement_factor": 1.0,
      "sequence_length": 5
    },
    "z_scores": {...},
    "density_metrics": {...},
    "anomaly_scores": {...},
    "normalized_sequence": [0.0, 0.25, 0.5, 0.75, 1.0]
  }
}
```

#### GET /api/info
Get API information and supported data types.

**Response:**
```json
{
  "service": "Universal Invariant Scoring Engine",
  "version": "1.0.0",
  "description": "Z-invariant based scoring and density analysis for arbitrary sequences",
  "supported_data_types": [...],
  "endpoints": [...]
}
```

For complete documentation, examples, and mathematical background, see the full API documentation.