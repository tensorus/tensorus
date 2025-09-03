# Getting Started with Tensorus

## Welcome to Tensorus

Tensorus is the world's first production-ready tensor database, purpose-built for high-performance machine learning and AI applications. This guide will get you up and running in minutes.

## What You'll Learn

- Install and configure Tensorus
- Store your first tensor
- Execute tensor operations
- Query tensor data with NQL
- Monitor performance and health

**Time to Complete**: 15-20 minutes

## Prerequisites

- **Python**: Version 3.8+ (3.10+ recommended)
- **Operating System**: Linux, macOS, or Windows with WSL2
- **Memory**: 4+ GB available RAM
- **Storage**: 10+ GB available disk space

## Quick Start (5 Minutes)

### 1. Install Tensorus

```bash
# Install via pip
pip install tensorus

# Or install with development extras
pip install tensorus[dev,gpu]

# Verify installation
tensorus --version
```

### 2. Start Tensorus Server

```bash
# Start in development mode
tensorus start --dev

# Server will be available at http://localhost:8000
# API documentation at http://localhost:8000/docs
```

### 3. Store Your First Tensor

```python
import tensorus
import numpy as np

# Initialize client
client = tensorus.Client()

# Create a dataset
client.create_dataset("my_first_dataset")

# Create and store a tensor
my_tensor = np.random.random((10, 10))
tensor_id = client.store_tensor(
    dataset="my_first_dataset",
    tensor=my_tensor,
    metadata={"type": "random", "purpose": "demo"}
)

print(f"Stored tensor with ID: {tensor_id}")
```

### 4. Query Your Data

```python
# Retrieve tensor by ID
retrieved = client.get_tensor(tensor_id)
print(f"Retrieved tensor shape: {retrieved.shape}")

# Query with Natural Language
results = client.query("find tensors where metadata.type = 'random'")
print(f"Found {len(results)} matching tensors")
```

### 5. Execute Operations

```python
# Perform tensor operations
result = client.matrix_multiply(tensor_id, tensor_id)
print(f"Matrix multiplication result shape: {result.shape}")

# Check operation history
history = client.get_operation_history()
print(f"Total operations: {len(history)}")
```

**🎉 Congratulations!** You've successfully stored, queried, and operated on tensors with Tensorus.

## Detailed Installation

### System Requirements

#### Minimum Requirements
- **CPU**: 2 cores (4 threads)
- **Memory**: 4 GB RAM
- **Storage**: 10 GB available space
- **Python**: 3.8+

#### Recommended for Development
- **CPU**: 4+ cores (8+ threads)  
- **Memory**: 16+ GB RAM
- **Storage**: 100+ GB SSD
- **Python**: 3.10+

#### Production Requirements
See [Production Deployment Guide](production_deployment.md) for detailed specifications.

### Installation Options

#### Option 1: Standard Installation (Recommended)

```bash
# Install stable version
pip install tensorus

# Install with optional dependencies
pip install tensorus[compression,gpu,monitoring]

# Verify installation
python -c "import tensorus; print(tensorus.__version__)"
```

#### Option 2: Development Installation

```bash
# Clone repository
git clone https://github.com/tensorus/tensorus.git
cd tensorus

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .[dev,test]

# Run tests to verify
python -m pytest tests/ -v
```

#### Option 3: Docker Installation

```bash
# Pull Docker image
docker pull tensorus/tensorus:latest

# Run container
docker run -d \
  --name tensorus \
  -p 8000:8000 \
  -v tensorus_data:/app/data \
  tensorus/tensorus:latest

# Verify container is running
docker ps
curl http://localhost:8000/health
```

### Configuration

#### Basic Configuration

Create a configuration file at `~/.tensorus/config.yml`:

```yaml
# ~/.tensorus/config.yml
environment: development

api:
  host: "127.0.0.1"
  port: 8000
  
database:
  url: "sqlite:///~/.tensorus/tensorus.db"
  
storage:
  path: "~/.tensorus/data"
  compression: "balanced"
  
logging:
  level: "INFO"
  file: "~/.tensorus/tensorus.log"
```

#### Environment Variables

```bash
# Optional configuration via environment variables
export TENSORUS_API_HOST="0.0.0.0"
export TENSORUS_API_PORT="8000"
export TENSORUS_LOG_LEVEL="DEBUG"
export TENSORUS_STORAGE_PATH="/opt/tensorus/data"
```

## Complete Tutorial: Building Your First AI Application

### Scenario: Image Classification Pipeline

We'll build a complete pipeline for storing, processing, and analyzing image tensors.

#### Step 1: Setup and Data Preparation

```python
import tensorus
import numpy as np
import torch
from PIL import Image
import requests
from io import BytesIO

# Initialize Tensorus client
client = tensorus.Client()

# Create datasets for our pipeline
client.create_dataset("raw_images", description="Original image data")
client.create_dataset("preprocessed", description="Preprocessed image tensors")
client.create_dataset("features", description="Extracted features")
client.create_dataset("predictions", description="Model predictions")

print("✅ Datasets created successfully")
```

#### Step 2: Load and Store Image Data

```python
def load_sample_images():
    """Load sample images from URLs"""
    sample_urls = [
        "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/5/50/Vd-Orig.png/256px-Vd-Orig.png"
    ]
    
    image_tensors = []
    for i, url in enumerate(sample_urls):
        try:
            # Download image
            response = requests.get(url)
            img = Image.open(BytesIO(response.content))
            
            # Convert to tensor
            img_array = np.array(img.convert('RGB')).astype(np.float32) / 255.0
            
            # Store in Tensorus
            tensor_id = client.store_tensor(
                dataset="raw_images",
                tensor=img_array,
                metadata={
                    "source_url": url,
                    "image_index": i,
                    "format": "RGB",
                    "normalized": True
                }
            )
            
            image_tensors.append(tensor_id)
            print(f"✅ Stored image {i+1} with ID: {tensor_id}")
            
        except Exception as e:
            print(f"❌ Failed to load image {i+1}: {e}")
    
    return image_tensors

# Load sample data
image_ids = load_sample_images()
```

#### Step 3: Preprocess Images

```python
def preprocess_images(image_ids):
    """Preprocess raw images for ML model"""
    preprocessed_ids = []
    
    for image_id in image_ids:
        # Retrieve original image
        image_tensor = client.get_tensor(image_id)
        
        # Resize to standard size (224x224 for many models)
        resized = torch.nn.functional.interpolate(
            torch.from_numpy(image_tensor).unsqueeze(0).permute(0, 3, 1, 2),
            size=(224, 224),
            mode='bilinear',
            align_corners=False
        )
        
        # Normalize for pretrained models (ImageNet stats)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        normalized = (resized - mean) / std
        
        # Store preprocessed tensor
        preprocessed_id = client.store_tensor(
            dataset="preprocessed",
            tensor=normalized.squeeze(0).numpy(),
            metadata={
                "original_id": image_id,
                "preprocessing": "resized_224_normalized",
                "ready_for_inference": True
            }
        )
        
        preprocessed_ids.append(preprocessed_id)
        print(f"✅ Preprocessed image: {preprocessed_id}")
    
    return preprocessed_ids

# Preprocess all images
preprocessed_ids = preprocess_images(image_ids)
```

#### Step 4: Extract Features (Simulated)

```python
def extract_features(preprocessed_ids):
    """Extract features using a simulated CNN model"""
    feature_ids = []
    
    for prep_id in preprocessed_ids:
        # Get preprocessed tensor
        prep_tensor = client.get_tensor(prep_id)
        
        # Simulate feature extraction (replace with real model)
        # For demo, we'll create random features
        features = np.random.random((512,))  # Typical CNN feature size
        
        # Store features
        feature_id = client.store_tensor(
            dataset="features",
            tensor=features,
            metadata={
                "source_preprocessed_id": prep_id,
                "feature_extractor": "simulated_cnn",
                "feature_dim": 512
            }
        )
        
        feature_ids.append(feature_id)
        print(f"✅ Extracted features: {feature_id}")
    
    return feature_ids

# Extract features
feature_ids = extract_features(preprocessed_ids)
```

#### Step 5: Query and Analyze Data

```python
# Query examples using Natural Language Query (NQL)
print("\n🔍 Querying stored data:")

# Find all preprocessed images
results = client.query("find tensors from 'preprocessed' where metadata.ready_for_inference = true")
print(f"✅ Found {len(results)} preprocessed images")

# Find features with specific dimensions
results = client.query("find tensors from 'features' where metadata.feature_dim = 512")
print(f"✅ Found {len(results)} feature vectors")

# Find original images by format
results = client.query("find tensors from 'raw_images' where metadata.format = 'RGB'")
print(f"✅ Found {len(results)} RGB images")

# Advanced query: Find processing pipeline
results = client.query("""
    find tensors where metadata contains 'original_id' 
    and dataset = 'preprocessed'
""")
print(f"✅ Found {len(results)} processed images with source tracking")
```

#### Step 6: Perform Tensor Operations

```python
print("\n⚙️ Executing tensor operations:")

if len(feature_ids) >= 2:
    # Compute similarity between feature vectors
    similarity = client.cosine_similarity(feature_ids[0], feature_ids[1])
    print(f"✅ Similarity between features: {similarity:.4f}")
    
    # Compute feature average
    avg_features = client.mean([feature_ids[0], feature_ids[1]])
    print(f"✅ Computed average features, shape: {avg_features.shape}")
    
    # Element-wise operations
    feature_sum = client.add(feature_ids[0], feature_ids[1])
    print(f"✅ Added feature vectors, shape: {feature_sum.shape}")
```

#### Step 7: Monitor and Analyze Performance

```python
print("\n📊 Performance Analysis:")

# Get operation history
history = client.get_operation_history(limit=10)
print(f"✅ Recent operations: {len(history)}")

for op in history[:3]:  # Show last 3 operations
    print(f"  - {op['operation_type']}: {op['duration_ms']}ms")

# Get dataset statistics
for dataset_name in ["raw_images", "preprocessed", "features"]:
    stats = client.get_dataset_stats(dataset_name)
    print(f"✅ {dataset_name}: {stats['tensor_count']} tensors, {stats['total_size_mb']:.1f} MB")

# Check storage efficiency
compression_stats = client.get_compression_stats()
print(f"✅ Compression ratio: {compression_stats['average_ratio']:.2f}x")
print(f"✅ Space saved: {compression_stats['space_saved_percent']:.1f}%")
```

#### Step 8: Visualize Data Lineage

```python
print("\n🔗 Data Lineage Tracking:")

# Trace lineage for a feature vector
if feature_ids:
    lineage = client.get_tensor_lineage(feature_ids[0])
    print(f"✅ Lineage depth: {lineage['max_depth']} levels")
    print(f"✅ Total operations: {lineage['total_operations']}")
    
    # Show the processing chain
    print("Processing chain:")
    for node in lineage['lineage_nodes'][:3]:
        print(f"  - {node['tensor_id'][:8]}... (depth {node['depth']})")
```

### Complete Tutorial Output

When you run the complete tutorial, you should see output like:

```
✅ Datasets created successfully
✅ Stored image 1 with ID: 550e8400-e29b-41d4-a716-446655440001
✅ Stored image 2 with ID: 550e8400-e29b-41d4-a716-446655440002
✅ Preprocessed image: 550e8400-e29b-41d4-a716-446655440003
✅ Preprocessed image: 550e8400-e29b-41d4-a716-446655440004
✅ Extracted features: 550e8400-e29b-41d4-a716-446655440005
✅ Extracted features: 550e8400-e29b-41d4-a716-446655440006

🔍 Querying stored data:
✅ Found 2 preprocessed images
✅ Found 2 feature vectors
✅ Found 2 RGB images
✅ Found 2 processed images with source tracking

⚙️ Executing tensor operations:
✅ Similarity between features: 0.3421
✅ Computed average features, shape: (512,)
✅ Added feature vectors, shape: (512,)

📊 Performance Analysis:
✅ Recent operations: 8
  - cosine_similarity: 15ms
  - mean: 8ms
  - add: 5ms
✅ raw_images: 2 tensors, 2.1 MB
✅ preprocessed: 2 tensors, 1.8 MB
✅ features: 2 tensors, 0.004 MB
✅ Compression ratio: 2.15x
✅ Space saved: 53.5%

🔗 Data Lineage Tracking:
✅ Lineage depth: 3 levels
✅ Total operations: 5
Processing chain:
  - 550e8400... (depth 0)
  - 550e8400... (depth 1)
  - 550e8400... (depth 2)
```

## Web Interface Tour

### Accessing the Dashboard

1. Start Tensorus: `tensorus start --dev`
2. Open browser: `http://localhost:8000`
3. Navigate to different sections:

#### Dashboard Overview
- System health and performance metrics
- Recent operations and activities
- Storage utilization and statistics

#### Data Explorer
- Browse datasets and tensors
- View tensor metadata and properties
- Visualize tensor shapes and distributions

#### NQL Chatbot
- Interactive natural language queries
- Query suggestions and examples
- Real-time result visualization

#### API Playground  
- Interactive API documentation
- Test endpoints with live data
- Generate code samples

#### Control Panel
- Server configuration and settings
- User management and permissions
- System monitoring and logs

## Python SDK Deep Dive

### Client Configuration

```python
import tensorus

# Basic client
client = tensorus.Client()

# Configured client
client = tensorus.Client(
    api_key="your-api-key",
    base_url="http://localhost:8000",
    timeout=30,
    max_retries=3
)

# Enterprise client with advanced features
client = tensorus.EnterpriseClient(
    api_key="enterprise-key",
    base_url="https://api.tensorus.com",
    organization_id="your-org-id",
    enable_caching=True,
    cache_size_mb=512
)
```

### Advanced Operations

```python
# Batch operations
results = client.batch_operations([
    {"operation": "normalize", "inputs": {"tensor": tensor_id}},
    {"operation": "reduce_mean", "inputs": {"tensor": "@previous", "axis": 0}},
    {"operation": "softmax", "inputs": {"tensor": "@previous"}}
])

# Asynchronous operations for large tensors
job = client.async_operation(
    operation="svd_decomposition",
    inputs={"tensor": large_tensor_id},
    parameters={"rank": 50}
)

# Monitor job progress
while not job.is_complete():
    print(f"Progress: {job.progress}%")
    time.sleep(1)

result = job.get_result()
```

### Error Handling

```python
import tensorus
from tensorus.exceptions import TensorNotFoundError, OperationFailedError

try:
    tensor = client.get_tensor("invalid-id")
except TensorNotFoundError:
    print("Tensor not found")
except OperationFailedError as e:
    print(f"Operation failed: {e.message}")
except tensorus.TensorusError as e:
    print(f"General error: {e}")
```

## Common Use Cases

### 1. ML Model Training Pipeline

```python
# Store training data
for batch in training_data_loader:
    batch_id = client.store_tensor(
        dataset="training_batches",
        tensor=batch,
        metadata={"epoch": current_epoch, "batch_size": len(batch)}
    )

# Track model checkpoints
checkpoint_id = client.store_tensor(
    dataset="model_checkpoints",
    tensor=model.state_dict(),
    metadata={
        "epoch": epoch,
        "loss": train_loss,
        "accuracy": train_accuracy,
        "optimizer": "adam",
        "learning_rate": 0.001
    }
)

# Find best checkpoint
best_checkpoints = client.query("""
    find tensors from 'model_checkpoints' 
    where metadata.accuracy > 0.95
    order by metadata.accuracy desc
    limit 5
""")
```

### 2. Computer Vision Pipeline

```python
# Store and process image collections
image_dataset = "surveillance_footage"
client.create_dataset(image_dataset)

for frame in video_frames:
    # Store original frame
    frame_id = client.store_tensor(
        dataset=image_dataset,
        tensor=frame,
        metadata={"timestamp": timestamp, "camera_id": camera_id}
    )
    
    # Apply object detection
    detections = client.apply_operation(
        operation="object_detection",
        tensor_id=frame_id,
        model="yolo_v8"
    )
    
    # Store detection results
    client.store_tensor(
        dataset="detections",
        tensor=detections,
        metadata={
            "source_frame": frame_id,
            "detection_count": len(detections),
            "confidence_threshold": 0.8
        }
    )
```

### 3. Time Series Analysis

```python
# Store sensor data streams
sensor_data = client.create_dataset("iot_sensors")

for sensor_reading in data_stream:
    reading_id = client.store_tensor(
        dataset="iot_sensors",
        tensor=sensor_reading.values,
        metadata={
            "sensor_id": sensor_reading.sensor_id,
            "timestamp": sensor_reading.timestamp,
            "location": sensor_reading.location,
            "measurement_type": sensor_reading.type
        }
    )

# Analyze patterns
recent_data = client.query("""
    find tensors from 'iot_sensors'
    where metadata.timestamp > '2024-01-01'
    and metadata.sensor_id = 'temperature_01'
""")

# Compute moving averages
for tensor_id in recent_data:
    smoothed = client.moving_average(tensor_id, window_size=10)
```

## Next Steps

### Explore Advanced Features

1. **[Performance Optimization](performance_benchmarks.md)** - Optimize your deployment for maximum performance
2. **[Production Deployment](production_deployment.md)** - Deploy Tensorus in production environments
3. **[API Reference](api_reference.md)** - Complete API documentation
4. **[Security Guide](security_compliance.md)** - Secure your Tensorus deployment

### Join the Community

- **GitHub**: [github.com/tensorus/tensorus](https://github.com/tensorus/tensorus)
- **Discord**: [discord.gg/tensorus](https://discord.gg/tensorus)  
- **Forum**: [community.tensorus.com](https://community.tensorus.com)
- **Stack Overflow**: Tag your questions with `tensorus`

### Get Support

- **Documentation**: [docs.tensorus.com](https://docs.tensorus.com)
- **Tutorials**: [learn.tensorus.com](https://learn.tensorus.com)
- **Support Portal**: [support.tensorus.com](https://support.tensorus.com)
- **Enterprise Support**: enterprise@tensorus.com

## Troubleshooting

### Installation Issues

**Problem**: `pip install tensorus` fails
```bash
# Solution: Update pip and use verbose mode
pip install --upgrade pip
pip install -v tensorus
```

**Problem**: ImportError after installation
```python
# Solution: Check Python path and reinstall
import sys
print(sys.path)
pip uninstall tensorus
pip install tensorus
```

### Runtime Issues

**Problem**: Connection refused when starting server
```bash
# Check if port is in use
netstat -tulpn | grep 8000
# Use different port
tensorus start --port 8080
```

**Problem**: Out of memory errors
```python
# Solution: Enable compression and chunking
client = tensorus.Client(config={
    "compression": "balanced",
    "chunking": {"enable": True, "size_mb": 64}
})
```

### Getting Help

If you encounter issues:

1. **Check the logs**: `~/.tensorus/tensorus.log`
2. **Search documentation**: [docs.tensorus.com](https://docs.tensorus.com)
3. **Ask the community**: [community.tensorus.com](https://community.tensorus.com)
4. **File a bug report**: [github.com/tensorus/tensorus/issues](https://github.com/tensorus/tensorus/issues)

---

**Welcome to the future of tensor data management! 🚀**

*Get started today and experience the power of purpose-built tensor databases.*