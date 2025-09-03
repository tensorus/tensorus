# Tensorus Examples and Practical Guides

This directory contains comprehensive examples, tutorials, and guides for using Tensorus effectively. These materials address **GAP 9: Limited Practical Examples** by providing real-world usage scenarios, complete implementations, and best practices.

## 📚 Contents Overview

### Core Examples
- **`compression_demo.py`** - Compression and quantization capabilities
- **`storage_ops_demo.py`** - Storage-connected tensor operations
- **`tensor_operations_comprehensive.py`** - Complete tensor operations reference
- **`real_world_tutorials.py`** - Machine learning, time series, and image processing workflows
- **`api_integration_examples.py`** - REST API client implementations
- **`use_case_guides.py`** - Data science pipelines and production optimization

## 🎯 Quick Start Guide

### Basic Usage Example
```python
from tensorus.tensor_storage import TensorStorage
from tensorus.tensor_ops import TensorOps
import torch

# Initialize storage
storage = TensorStorage("./my_tensorus_data")
storage.create_dataset("my_dataset")

# Store a tensor
tensor = torch.randn(100, 50)
metadata = {"name": "sample_data", "type": "features"}
tensor_id = storage.insert("my_dataset", tensor, metadata)

# Perform operations
result = TensorOps.add(tensor, 5.0)
print(f"Operation result shape: {result.shape}")
```

### Running Examples

1. **Basic tensor operations:**
   ```bash
   python tensor_operations_comprehensive.py
   ```

2. **Real-world tutorials:**
   ```bash
   python real_world_tutorials.py
   ```

3. **API integration examples:**
   ```bash
   python api_integration_examples.py
   ```

## 📖 Detailed Example Descriptions

### 1. Tensor Operations Comprehensive (`tensor_operations_comprehensive.py`)

**Purpose:** Complete reference for all available tensor operations in Tensorus.

**What you'll learn:**
- All arithmetic operations (add, subtract, multiply, divide, power, log)
- Matrix operations (matmul, dot, outer, cross products)  
- Reduction operations (sum, mean, min, max across dimensions)
- Reshaping operations (reshape, transpose, permute, flatten)
- Advanced operations (Einstein summation, automatic differentiation)
- Linear algebra (eigendecomposition, SVD, QR, Cholesky)
- Convolution operations (1D, 2D, 3D)
- Statistical operations (variance, covariance, correlation, norms)
- Performance benchmarking techniques

**Key sections:**
- Basic arithmetic with error handling
- Matrix operations with shape validation
- Advanced linear algebra decompositions
- Statistical analysis and correlation
- Performance comparison across tensor sizes

### 2. Real-World Tutorials (`real_world_tutorials.py`)

**Purpose:** Complete workflows for common machine learning and data science tasks.

**Tutorials included:**

#### Tutorial 1: Machine Learning Model Management
- Neural network weight storage with compression
- Training checkpoint management across epochs
- Model optimization techniques (weight averaging)
- Performance analysis and gradient tracking
- Storage efficiency analysis

#### Tutorial 2: Time Series Data Processing  
- Multi-series synthetic data generation
- Preprocessing (normalization, smoothing, differencing)
- Feature extraction (technical indicators, Bollinger bands, RSI)
- Forecasting with AR models
- Cross-series correlation analysis

#### Tutorial 3: Image Processing Pipeline
- Synthetic dataset creation (geometric, noise, gradient patterns)
- Image preprocessing (normalization, histogram equalization, filtering)
- Feature extraction (histograms, LBP, edge density)
- Data augmentation pipeline
- Image similarity analysis

**Key features:**
- Complete end-to-end workflows
- Realistic data generation
- Production-ready error handling
- Performance monitoring
- Best practices demonstration

### 3. API Integration Examples (`api_integration_examples.py`)

**Purpose:** Complete REST API client implementations for different integration scenarios.

**Client implementations:**

#### Synchronous API Client
- Authentication management
- Error handling with retries
- Performance monitoring
- Request/response serialization
- Comprehensive tensor operations

#### Asynchronous API Client  
- High-performance concurrent operations
- Batch upload optimization
- Connection pooling
- Resource management

#### Production-Ready Client
- Exponential backoff retry logic
- Request analytics and monitoring
- Validation and error reporting
- Progress tracking for long operations

**Integration scenarios:**
- Basic API operations (health check, dataset management)
- Tensor operations via API
- Advanced querying and metadata filtering
- Production patterns and monitoring

### 4. Use Case Guides (`use_case_guides.py`)

**Purpose:** Detailed guides for specific deployment scenarios with best practices.

#### Guide 1: Data Science Pipeline Integration
- Complete workflow from ingestion to deployment
- Data quality checks and validation
- Feature engineering automation
- Model training integration
- Experiment tracking and comparison
- Pipeline monitoring and metrics

#### Guide 2: Production Performance Optimization
- Storage configuration optimization
- Compression strategy selection
- Batch processing optimization
- Memory management best practices
- Performance monitoring and alerting

**Key topics:**
- Configuration recommendations
- Performance benchmarking
- Resource management strategies
- Monitoring and alerting setup
- Troubleshooting guides

### 5. Storage Operations Demo (`storage_ops_demo.py`)

**Purpose:** Demonstrates storage-connected operations that work directly with tensor IDs.

**Features demonstrated:**
- Direct operations on stored tensors
- Result caching for performance
- Batch operations on multiple tensors
- Automatic metadata tracking
- Benchmarking and performance analysis
- Advanced operation chaining

### 6. Compression Demo (`compression_demo.py`)

**Purpose:** Shows compression and quantization capabilities for storage efficiency.

**Compression features:**
- Different algorithms (GZIP, LZ4)
- Quantization methods (INT8, FP16)
- Compression presets for easy configuration
- Performance vs. compression trade-offs
- Real-world usage scenarios

## 🛠️ Development and Testing

### Running All Examples
```bash
# Run individual examples
python tensor_operations_comprehensive.py
python real_world_tutorials.py
python api_integration_examples.py
python use_case_guides.py
python storage_ops_demo.py
python compression_demo.py

# Or use the test runner (if available)
pytest examples/ -v
```

### Dependencies
All examples use the core Tensorus modules:
- `tensorus.tensor_storage`
- `tensorus.tensor_ops` 
- `tensorus.storage_ops`
- `tensorus.compression`

Additional dependencies for specific examples:
- `matplotlib` - for plotting in some tutorials
- `numpy` - for numerical operations
- `requests` - for API client examples
- `aiohttp` - for async API client

## 🎯 Learning Path Recommendations

### For Beginners
1. Start with `tensor_operations_comprehensive.py` to understand basic operations
2. Run `compression_demo.py` to learn about storage efficiency
3. Try `storage_ops_demo.py` for advanced storage operations

### For Data Scientists
1. Focus on `real_world_tutorials.py` for complete workflows
2. Study the data science pipeline guide in `use_case_guides.py`
3. Learn API integration patterns from `api_integration_examples.py`

### For Production Deployments
1. Read production optimization guide in `use_case_guides.py`
2. Implement monitoring patterns from API integration examples
3. Test compression strategies with your actual data

### For Application Developers
1. Study API client implementations in `api_integration_examples.py`
2. Learn integration patterns from real-world tutorials
3. Implement error handling and retry logic from production examples

## 📊 Performance Considerations

### Benchmarking Results (Typical)
The examples include performance benchmarking code that demonstrates:

- **Matrix Operations:** 100x100 matrices process in ~1-5ms
- **Compression:** 2-15x compression ratios depending on data type
- **Batch Operations:** 10-15x throughput improvement with proper batching
- **API Calls:** <100ms latency for typical tensor operations

### Optimization Tips from Examples
1. **Use appropriate compression** - Balance storage vs. access speed
2. **Batch operations** - Significant performance gains for multiple tensors
3. **Monitor memory usage** - Especially important for large tensors
4. **Cache frequently accessed data** - Reduces repeated I/O operations
5. **Use async operations** - For high-throughput applications

## 🐛 Troubleshooting

### Common Issues and Solutions

**Issue: Out of memory errors**
- Solution: Use compression, implement batch processing with smaller chunks
- Example: See memory management in `use_case_guides.py`

**Issue: Slow tensor operations**
- Solution: Enable caching, use appropriate data types, batch operations
- Example: Performance optimization in `tensor_operations_comprehensive.py`

**Issue: API connection errors**  
- Solution: Implement retry logic, check authentication, verify endpoints
- Example: Error handling in `api_integration_examples.py`

**Issue: Storage growth**
- Solution: Use compression, implement data lifecycle management
- Example: Compression strategies in `compression_demo.py`

## 📈 Success Metrics

The examples demonstrate how to track:
- **Throughput:** Tensors processed per second
- **Storage Efficiency:** Compression ratios achieved
- **Reliability:** Success rates and error handling
- **Performance:** Operation latencies and resource usage

## 🔗 Related Documentation

- [API Guide](../docs/api_guide.md) - Complete API reference
- [Installation Guide](../docs/installation.md) - Setup and configuration
- [Metadata Schemas](../docs/metadata_schemas.md) - Data organization
- [Compression Implementation](../docs/compression_implementation.md) - Technical details

## 🤝 Contributing

To add new examples or improve existing ones:

1. Follow the existing code structure and documentation style
2. Include comprehensive error handling and logging
3. Add performance monitoring where appropriate
4. Provide clear explanations and use cases
5. Test with different data sizes and scenarios

## 📝 License

These examples are part of the Tensorus project and follow the same licensing terms.

---

**Note:** These examples provide comprehensive coverage of Tensorus capabilities and address the identified gap in practical examples. They serve as both learning resources and production-ready code templates for different use cases.