"""
Test suite for tensor compression and quantization functionality.
"""

import pytest
import torch
import numpy as np
from typing import Dict, Any
import tempfile
import shutil
from pathlib import Path

# Import compression modules
try:
    from tensorus.compression import (
        TensorCompression, CompressionConfig, get_compression_preset,
        GZIPCompression, LZ4Compression, NoCompression,
        INT8Quantization, FP16Quantization, NoQuantization,
        create_compression_algorithm, create_quantization_algorithm,
        CompressionError, QuantizationError
    )
    from tensorus.tensor_storage import TensorStorage
    COMPRESSION_AVAILABLE = True
except ImportError:
    COMPRESSION_AVAILABLE = False

# Skip all tests if compression module is not available
pytestmark = pytest.mark.skipif(not COMPRESSION_AVAILABLE, reason="Compression module not available")

class TestCompressionAlgorithms:
    """Test compression algorithms."""
    
    def test_no_compression(self):
        """Test NoCompression algorithm."""
        algo = NoCompression()
        data = b"test data"
        compressed = algo.compress(data)
        decompressed = algo.decompress(compressed)
        
        assert compressed == data
        assert decompressed == data
        assert algo.name == "none"
    
    def test_gzip_compression(self):
        """Test GZIP compression algorithm."""
        algo = GZIPCompression(compression_level=6)
        data = b"test data " * 100  # Repeated data compresses well
        compressed = algo.compress(data)
        decompressed = algo.decompress(compressed)
        
        assert len(compressed) < len(data)  # Should be smaller
        assert decompressed == data
        assert algo.name == "gzip-6"
    
    def test_lz4_compression(self):
        """Test LZ4 compression algorithm."""
        algo = LZ4Compression(compression_level=1)
        data = b"test data " * 100  # Repeated data compresses well
        compressed = algo.compress(data)
        decompressed = algo.decompress(compressed)
        
        assert len(compressed) < len(data)  # Should be smaller
        assert decompressed == data
        assert algo.name == "lz4-1"
    
    def test_compression_error(self):
        """Test compression error handling."""
        algo = GZIPCompression()
        
        # Test decompression of invalid data
        with pytest.raises(CompressionError):
            algo.decompress(b"invalid gzip data")

class TestQuantizationAlgorithms:
    """Test quantization algorithms."""
    
    def test_no_quantization(self):
        """Test NoQuantization algorithm."""
        algo = NoQuantization()
        tensor = torch.randn(10, 10)
        quantized, params = algo.quantize(tensor)
        dequantized = algo.dequantize(quantized, params)
        
        assert torch.equal(quantized, tensor)
        assert torch.equal(dequantized, tensor)
        assert algo.name == "none"
    
    def test_int8_quantization(self):
        """Test INT8 quantization algorithm."""
        algo = INT8Quantization()
        tensor = torch.randn(10, 10)
        quantized, params = algo.quantize(tensor)
        dequantized = algo.dequantize(quantized, params)
        
        assert quantized.dtype == torch.int8
        assert "scale" in params
        assert "zero_point" in params
        assert "original_dtype" in params
        assert algo.name == "int8"
        
        # Check approximate reconstruction
        assert torch.allclose(tensor, dequantized, rtol=0.1, atol=0.1)
    
    def test_fp16_quantization(self):
        """Test FP16 quantization algorithm."""
        algo = FP16Quantization()
        tensor = torch.randn(10, 10, dtype=torch.float32)
        quantized, params = algo.quantize(tensor)
        dequantized = algo.dequantize(quantized, params)
        
        assert quantized.dtype == torch.float16
        assert "original_dtype" in params
        assert algo.name == "fp16"
        
        # Check approximate reconstruction (FP16 has less precision)
        assert torch.allclose(tensor, dequantized, rtol=1e-3, atol=1e-3)
    
    def test_int8_constant_tensor(self):
        """Test INT8 quantization with constant tensor."""
        algo = INT8Quantization()
        tensor = torch.ones(5, 5) * 3.14
        quantized, params = algo.quantize(tensor)
        dequantized = algo.dequantize(quantized, params)
        
        assert quantized.dtype == torch.int8
        assert torch.allclose(tensor, dequantized, rtol=0.1, atol=0.1)

class TestTensorCompression:
    """Test TensorCompression class."""
    
    def test_no_compression_no_quantization(self):
        """Test with no compression and no quantization."""
        compressor = TensorCompression(
            NoCompression(),
            NoQuantization()
        )
        
        tensor = torch.randn(5, 5)
        compressed_bytes, metadata = compressor.compress_tensor(tensor)
        reconstructed = compressor.decompress_tensor(compressed_bytes, metadata)
        
        assert torch.equal(tensor, reconstructed)
        assert metadata["compression"] == "none"
        assert metadata["quantization"] == "none"
    
    def test_gzip_compression_int8_quantization(self):
        """Test with GZIP compression and INT8 quantization."""
        compressor = TensorCompression(
            GZIPCompression(),
            INT8Quantization()
        )
        
        tensor = torch.randn(20, 20)  # Larger tensor for better compression
        compressed_bytes, metadata = compressor.compress_tensor(tensor)
        reconstructed = compressor.decompress_tensor(compressed_bytes, metadata)
        
        assert metadata["compression"] == "gzip-6"
        assert metadata["quantization"] == "int8"
        assert metadata["compression_ratio"] > 1.0  # Should achieve some compression
        
        # Check approximate reconstruction
        assert torch.allclose(tensor, reconstructed, rtol=0.1, atol=0.1)
    
    def test_lz4_compression_fp16_quantization(self):
        """Test with LZ4 compression and FP16 quantization."""
        compressor = TensorCompression(
            LZ4Compression(),
            FP16Quantization()
        )
        
        tensor = torch.randn(15, 15)
        compressed_bytes, metadata = compressor.compress_tensor(tensor)
        reconstructed = compressor.decompress_tensor(compressed_bytes, metadata)
        
        assert metadata["compression"] == "lz4-1"
        assert metadata["quantization"] == "fp16"
        
        # Check approximate reconstruction (FP16 precision)
        assert torch.allclose(tensor, reconstructed, rtol=1e-3, atol=1e-3)

class TestCompressionConfig:
    """Test CompressionConfig class."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = CompressionConfig()
        assert config.compression == "none"
        assert config.quantization == "none"
        
        compressor = config.create_tensor_compression()
        assert compressor.compression_algorithm.name == "none"
        assert compressor.quantization_algorithm.name == "none"
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = CompressionConfig(
            compression="gzip",
            quantization="int8",
            compression_kwargs={"compression_level": 9},
            quantization_kwargs={}
        )
        
        compressor = config.create_tensor_compression()
        assert compressor.compression_algorithm.name == "gzip-9"
        assert compressor.quantization_algorithm.name == "int8"
    
    def test_from_dict(self):
        """Test creating config from dictionary."""
        config_dict = {
            "compression": "lz4",
            "quantization": "fp16",
            "compression_kwargs": {"compression_level": 2},
            "quantization_kwargs": {}
        }
        
        config = CompressionConfig.from_dict(config_dict)
        assert config.compression == "lz4"
        assert config.quantization == "fp16"
        assert config.compression_kwargs["compression_level"] == 2
    
    def test_to_dict(self):
        """Test converting config to dictionary."""
        config = CompressionConfig("gzip", "int8")
        config_dict = config.to_dict()
        
        assert config_dict["compression"] == "gzip"
        assert config_dict["quantization"] == "int8"

class TestCompressionPresets:
    """Test compression presets."""
    
    def test_available_presets(self):
        """Test that all presets are available."""
        presets = ["none", "fast", "balanced", "maximum", "fp16_only", "int8_only"]
        
        for preset in presets:
            config = get_compression_preset(preset)
            assert isinstance(config, CompressionConfig)
    
    def test_none_preset(self):
        """Test 'none' preset."""
        config = get_compression_preset("none")
        assert config.compression == "none"
        assert config.quantization == "none"
    
    def test_fast_preset(self):
        """Test 'fast' preset."""
        config = get_compression_preset("fast")
        assert config.compression == "lz4"
        assert config.quantization == "none"
    
    def test_balanced_preset(self):
        """Test 'balanced' preset."""
        config = get_compression_preset("balanced")
        assert config.compression == "gzip"
        assert config.quantization == "fp16"
    
    def test_maximum_preset(self):
        """Test 'maximum' preset."""
        config = get_compression_preset("maximum")
        assert config.compression == "gzip"
        assert config.quantization == "int8"
    
    def test_invalid_preset(self):
        """Test invalid preset name."""
        with pytest.raises(ValueError):
            get_compression_preset("invalid_preset")

class TestFactoryFunctions:
    """Test factory functions."""
    
    def test_create_compression_algorithm(self):
        """Test create_compression_algorithm function."""
        algo = create_compression_algorithm("gzip", compression_level=3)
        assert isinstance(algo, GZIPCompression)
        assert algo.name == "gzip-3"
        
        with pytest.raises(ValueError):
            create_compression_algorithm("invalid_algo")
    
    def test_create_quantization_algorithm(self):
        """Test create_quantization_algorithm function."""
        algo = create_quantization_algorithm("int8")
        assert isinstance(algo, INT8Quantization)
        assert algo.name == "int8"
        
        with pytest.raises(ValueError):
            create_quantization_algorithm("invalid_algo")

class TestTensorStorageIntegration:
    """Test integration with TensorStorage."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_tensor_storage_no_compression(self):
        """Test TensorStorage without compression."""
        storage = TensorStorage(
            storage_path=str(self.temp_path),
            compression_preset="none"
        )
        
        storage.create_dataset("test_dataset")
        tensor = torch.randn(10, 10)
        record_id = storage.insert("test_dataset", tensor, {"label": "test"})
        
        retrieved = storage.get_tensor_by_id("test_dataset", record_id)
        assert torch.equal(tensor, retrieved["tensor"])
        assert not retrieved["metadata"]["compressed"]
    
    def test_tensor_storage_with_compression(self):
        """Test TensorStorage with compression."""
        storage = TensorStorage(
            storage_path=str(self.temp_path),
            compression_preset="balanced"
        )
        
        storage.create_dataset("test_dataset")
        tensor = torch.randn(20, 20)  # Larger tensor for better compression
        record_id = storage.insert("test_dataset", tensor, {"label": "test"})
        
        retrieved = storage.get_tensor_by_id("test_dataset", record_id)
        assert retrieved["metadata"]["compressed"]
        assert "compression_metadata" in retrieved["metadata"]
        
        # Check approximate reconstruction (FP16 + compression)
        assert torch.allclose(tensor, retrieved["tensor"], rtol=1e-3, atol=1e-3)
    
    def test_tensor_storage_maximum_compression(self):
        """Test TensorStorage with maximum compression."""
        storage = TensorStorage(
            storage_path=str(self.temp_path),
            compression_preset="maximum"
        )
        
        storage.create_dataset("test_dataset")
        tensor = torch.randn(30, 30)  # Large tensor
        record_id = storage.insert("test_dataset", tensor, {"label": "test"})
        
        retrieved = storage.get_tensor_by_id("test_dataset", record_id)
        assert retrieved["metadata"]["compressed"]
        
        compression_meta = retrieved["metadata"]["compression_metadata"]
        assert compression_meta["compression"] == "gzip-9"
        assert compression_meta["quantization"] == "int8"
        assert compression_meta["compression_ratio"] > 1.0
        
        # Check approximate reconstruction (INT8 quantization is less precise)
        assert torch.allclose(tensor, retrieved["tensor"], rtol=0.1, atol=0.1)
    
    def test_tensor_storage_compression_stats(self):
        """Test compression statistics."""
        storage = TensorStorage(
            storage_path=str(self.temp_path),
            compression_preset="fast"
        )
        
        storage.create_dataset("test_dataset")
        
        # Insert compressed tensors
        for i in range(3):
            tensor = torch.randn(15, 15)
            storage.insert("test_dataset", tensor, {"label": f"test_{i}"})
        
        stats = storage.get_compression_stats("test_dataset")
        assert stats["total_tensors"] == 3
        assert stats["compressed_tensors"] == 3
        assert stats["total_original_size"] > 0
        assert stats["total_compressed_size"] > 0
        assert stats["average_compression_ratio"] > 1.0
        assert "lz4-1" in stats["compression_algorithms"]
    
    def test_tensor_storage_config_change(self):
        """Test changing compression configuration."""
        storage = TensorStorage(
            storage_path=str(self.temp_path),
            compression_preset="none"
        )
        
        storage.create_dataset("test_dataset")
        
        # Insert uncompressed tensor
        tensor1 = torch.randn(10, 10)
        record_id1 = storage.insert("test_dataset", tensor1)
        
        # Change to compression
        storage.set_compression_preset("fast")
        
        # Insert compressed tensor
        tensor2 = torch.randn(10, 10)
        record_id2 = storage.insert("test_dataset", tensor2)
        
        # Check both tensors
        retrieved1 = storage.get_tensor_by_id("test_dataset", record_id1)
        retrieved2 = storage.get_tensor_by_id("test_dataset", record_id2)
        
        assert not retrieved1["metadata"]["compressed"]
        assert retrieved2["metadata"]["compressed"]
        
        assert torch.equal(tensor1, retrieved1["tensor"])
        assert torch.allclose(tensor2, retrieved2["tensor"], rtol=1e-6, atol=1e-6)
    
    def test_tensor_storage_persistence_with_compression(self):
        """Test persistence with compression."""
        # Create storage with compression
        storage1 = TensorStorage(
            storage_path=str(self.temp_path),
            compression_preset="balanced"
        )
        
        storage1.create_dataset("test_dataset")
        tensor = torch.randn(20, 20)
        record_id = storage1.insert("test_dataset", tensor, {"label": "test"})
        
        # Create new storage instance (loads from disk)
        storage2 = TensorStorage(
            storage_path=str(self.temp_path),
            compression_preset="balanced"
        )
        
        # Check that compressed data was properly loaded
        retrieved = storage2.get_tensor_by_id("test_dataset", record_id)
        assert retrieved["metadata"]["compressed"]
        assert torch.allclose(tensor, retrieved["tensor"], rtol=1e-3, atol=1e-3)
    
    def test_tensor_storage_query_with_compression(self):
        """Test querying dataset with compressed tensors."""
        storage = TensorStorage(
            storage_path=str(self.temp_path),
            compression_preset="fp16_only"
        )
        
        storage.create_dataset("test_dataset")
        
        # Insert tensors with different values
        tensors = []
        for i in range(5):
            tensor = torch.ones(5, 5) * i
            storage.insert("test_dataset", tensor, {"value": i})
            tensors.append(tensor)
        
        # Query for tensors with mean > 2
        results = storage.query("test_dataset", lambda t, m: t.mean().item() > 2)
        
        assert len(results) == 2  # Should find tensors with values 3 and 4
        for result in results:
            assert result["tensor"].mean().item() > 2
    
    def test_tensor_storage_sampling_with_compression(self):
        """Test sampling dataset with compressed tensors."""
        storage = TensorStorage(
            storage_path=str(self.temp_path),
            compression_preset="int8_only"
        )
        
        storage.create_dataset("test_dataset")
        
        # Insert multiple tensors
        for i in range(10):
            tensor = torch.randn(5, 5)
            storage.insert("test_dataset", tensor, {"index": i})
        
        # Sample 3 tensors
        samples = storage.sample_dataset("test_dataset", 3)
        
        assert len(samples) == 3
        for sample in samples:
            assert "tensor" in sample
            assert "metadata" in sample
            assert sample["metadata"]["compressed"]

class TestErrorHandling:
    """Test error handling in compression system."""
    
    def test_compression_without_module(self):
        """Test behavior when compression module is not available."""
        # This would require mocking the import failure
        pass
    
    def test_invalid_compressed_data(self):
        """Test handling of invalid compressed data."""
        compressor = TensorCompression(
            GZIPCompression(),
            NoQuantization()
        )
        
        # Try to decompress invalid data
        with pytest.raises(CompressionError):
            compressor.decompress_tensor(b"invalid data", {
                "compression": "gzip-6",
                "quantization": "none",
                "quantization_params": {}
            })

if __name__ == "__main__":
    pytest.main([__file__])