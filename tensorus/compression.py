"""
Tensorus Compression and Quantization Module.

Provides compression and quantization support for tensor storage to reduce
memory usage and storage requirements.
"""

import torch
import gzip
import lz4.frame
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union, Tuple
from io import BytesIO
import numpy as np

logger = logging.getLogger(__name__)

class CompressionError(Exception):
    """Raised when compression/decompression operations fail."""
    pass

class QuantizationError(Exception):
    """Raised when quantization/dequantization operations fail."""
    pass

# === Compression Interface ===

class CompressionAlgorithm(ABC):
    """Abstract base class for compression algorithms."""
    
    @abstractmethod
    def compress(self, data: bytes) -> bytes:
        """Compress raw bytes."""
        pass
    
    @abstractmethod
    def decompress(self, compressed_data: bytes) -> bytes:
        """Decompress raw bytes."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return algorithm name."""
        pass

class GZIPCompression(CompressionAlgorithm):
    """GZIP compression algorithm."""
    
    def __init__(self, compression_level: int = 6):
        """Initialize with compression level (1-9)."""
        self.compression_level = max(1, min(9, compression_level))
    
    def compress(self, data: bytes) -> bytes:
        """Compress data using GZIP."""
        try:
            return gzip.compress(data, compresslevel=self.compression_level)
        except Exception as e:
            raise CompressionError(f"GZIP compression failed: {e}")
    
    def decompress(self, compressed_data: bytes) -> bytes:
        """Decompress GZIP data."""
        try:
            return gzip.decompress(compressed_data)
        except Exception as e:
            raise CompressionError(f"GZIP decompression failed: {e}")
    
    @property
    def name(self) -> str:
        return f"gzip-{self.compression_level}"

class LZ4Compression(CompressionAlgorithm):
    """LZ4 compression algorithm."""
    
    def __init__(self, compression_level: int = 1):
        """Initialize with compression level."""
        self.compression_level = compression_level
    
    def compress(self, data: bytes) -> bytes:
        """Compress data using LZ4."""
        try:
            return lz4.frame.compress(
                data, 
                compression_level=self.compression_level
            )
        except Exception as e:
            raise CompressionError(f"LZ4 compression failed: {e}")
    
    def decompress(self, compressed_data: bytes) -> bytes:
        """Decompress LZ4 data."""
        try:
            return lz4.frame.decompress(compressed_data)
        except Exception as e:
            raise CompressionError(f"LZ4 decompression failed: {e}")
    
    @property
    def name(self) -> str:
        return f"lz4-{self.compression_level}"

class NoCompression(CompressionAlgorithm):
    """No compression (pass-through)."""
    
    def compress(self, data: bytes) -> bytes:
        return data
    
    def decompress(self, compressed_data: bytes) -> bytes:
        return compressed_data
    
    @property
    def name(self) -> str:
        return "none"

# === Quantization Interface ===

class QuantizationAlgorithm(ABC):
    """Abstract base class for quantization algorithms."""
    
    @abstractmethod
    def quantize(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Quantize tensor, return quantized tensor and quantization parameters."""
        pass
    
    @abstractmethod
    def dequantize(self, quantized_tensor: torch.Tensor, params: Dict[str, Any]) -> torch.Tensor:
        """Dequantize tensor using parameters."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return algorithm name."""
        pass

class INT8Quantization(QuantizationAlgorithm):
    """INT8 quantization using linear mapping."""
    
    def quantize(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Quantize to INT8 using linear mapping."""
        try:
            if tensor.dtype == torch.int8:
                logger.debug("Tensor already INT8 quantized")
                return tensor, {"scale": 1.0, "zero_point": 0, "original_dtype": str(tensor.dtype)}
            
            # Calculate scale and zero point
            min_val = tensor.min().item()
            max_val = tensor.max().item()
            
            if min_val == max_val:
                # Handle constant tensors
                scale = 1.0
                zero_point = int(min_val)  # Use the constant value as zero_point
                zero_point = max(-128, min(127, zero_point))
                quantized = torch.full_like(tensor, zero_point, dtype=torch.int8)
            else:
                # Map to [-128, 127] range
                scale = (max_val - min_val) / 255.0
                zero_point = int(-128 - min_val / scale)
                zero_point = max(-128, min(127, zero_point))
                
                quantized = torch.clamp(
                    torch.round(tensor / scale) + zero_point,
                    -128, 127
                ).to(torch.int8)
            
            params = {
                "scale": scale,
                "zero_point": zero_point,
                "original_dtype": str(tensor.dtype),
                "original_shape": list(tensor.shape)
            }
            
            return quantized, params
            
        except Exception as e:
            raise QuantizationError(f"INT8 quantization failed: {e}")
    
    def dequantize(self, quantized_tensor: torch.Tensor, params: Dict[str, Any]) -> torch.Tensor:
        """Dequantize INT8 tensor."""
        try:
            scale = params["scale"]
            zero_point = params["zero_point"]
            original_dtype = params["original_dtype"]
            
            # Convert back to float
            if scale == 1.0 and isinstance(zero_point, (int, float)):
                # Handle constant tensors - special case
                dequantized = torch.full_like(quantized_tensor, zero_point, dtype=torch.float32)
            else:
                # Normal quantization case
                dequantized = (quantized_tensor.to(torch.float32) - zero_point) * scale
            
            # Convert to original dtype
            if original_dtype != "torch.int8":
                target_dtype = getattr(torch, original_dtype.split('.')[1])
                dequantized = dequantized.to(target_dtype)
            
            return dequantized
            
        except Exception as e:
            raise QuantizationError(f"INT8 dequantization failed: {e}")
    
    @property
    def name(self) -> str:
        return "int8"

class FP16Quantization(QuantizationAlgorithm):
    """FP16 quantization (half precision)."""
    
    def quantize(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Quantize to FP16."""
        try:
            if tensor.dtype == torch.float16:
                logger.debug("Tensor already FP16 quantized")
                return tensor, {"original_dtype": str(tensor.dtype)}
            
            quantized = tensor.to(torch.float16)
            params = {
                "original_dtype": str(tensor.dtype),
                "original_shape": list(tensor.shape)
            }
            
            return quantized, params
            
        except Exception as e:
            raise QuantizationError(f"FP16 quantization failed: {e}")
    
    def dequantize(self, quantized_tensor: torch.Tensor, params: Dict[str, Any]) -> torch.Tensor:
        """Dequantize FP16 tensor."""
        try:
            original_dtype = params["original_dtype"]
            
            if original_dtype == "torch.float16":
                return quantized_tensor
            
            # Convert to original dtype
            target_dtype = getattr(torch, original_dtype.split('.')[1])
            return quantized_tensor.to(target_dtype)
            
        except Exception as e:
            raise QuantizationError(f"FP16 dequantization failed: {e}")
    
    @property
    def name(self) -> str:
        return "fp16"

class NoQuantization(QuantizationAlgorithm):
    """No quantization (pass-through)."""
    
    def quantize(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        return tensor, {"original_dtype": str(tensor.dtype)}
    
    def dequantize(self, quantized_tensor: torch.Tensor, params: Dict[str, Any]) -> torch.Tensor:
        return quantized_tensor
    
    @property
    def name(self) -> str:
        return "none"

# === Compression Manager ===

class TensorCompression:
    """Manages tensor compression and quantization."""
    
    def __init__(self, 
                 compression_algorithm: Optional[CompressionAlgorithm] = None,
                 quantization_algorithm: Optional[QuantizationAlgorithm] = None):
        """Initialize with compression and quantization algorithms."""
        self.compression_algorithm = compression_algorithm or NoCompression()
        self.quantization_algorithm = quantization_algorithm or NoQuantization()
        
    def compress_tensor(self, tensor: torch.Tensor) -> Tuple[bytes, Dict[str, Any]]:
        """Compress a tensor with quantization and compression."""
        try:
            # Step 1: Quantize
            quantized_tensor, quant_params = self.quantization_algorithm.quantize(tensor)
            
            # Step 2: Serialize to bytes
            buffer = BytesIO()
            torch.save(quantized_tensor, buffer)
            tensor_bytes = buffer.getvalue()
            
            # Step 3: Compress
            compressed_bytes = self.compression_algorithm.compress(tensor_bytes)
            
            # Create metadata
            metadata = {
                "compression": self.compression_algorithm.name,
                "quantization": self.quantization_algorithm.name,
                "quantization_params": quant_params,
                "original_size": len(tensor_bytes),
                "compressed_size": len(compressed_bytes),
                "compression_ratio": len(tensor_bytes) / len(compressed_bytes) if compressed_bytes else 1.0
            }
            
            logger.debug(f"Tensor compressed: {metadata['original_size']} -> {metadata['compressed_size']} bytes "
                        f"(ratio: {metadata['compression_ratio']:.2f}x)")
            
            return compressed_bytes, metadata
            
        except Exception as e:
            raise CompressionError(f"Tensor compression failed: {e}")
    
    def decompress_tensor(self, compressed_bytes: bytes, metadata: Dict[str, Any]) -> torch.Tensor:
        """Decompress a tensor."""
        try:
            # Step 1: Decompress
            tensor_bytes = self.compression_algorithm.decompress(compressed_bytes)
            
            # Step 2: Deserialize from bytes
            buffer = BytesIO(tensor_bytes)
            quantized_tensor = torch.load(buffer, map_location="cpu")
            
            # Step 3: Dequantize
            quant_params = metadata.get("quantization_params", {})
            tensor = self.quantization_algorithm.dequantize(quantized_tensor, quant_params)
            
            return tensor
            
        except Exception as e:
            raise CompressionError(f"Tensor decompression failed: {e}")

# === Factory Functions ===

def create_compression_algorithm(name: str, **kwargs) -> CompressionAlgorithm:
    """Create compression algorithm by name."""
    algorithms = {
        "gzip": GZIPCompression,
        "lz4": LZ4Compression,
        "none": NoCompression
    }
    
    if name not in algorithms:
        raise ValueError(f"Unknown compression algorithm: {name}. Available: {list(algorithms.keys())}")
    
    return algorithms[name](**kwargs)

def create_quantization_algorithm(name: str, **kwargs) -> QuantizationAlgorithm:
    """Create quantization algorithm by name."""
    algorithms = {
        "int8": INT8Quantization,
        "fp16": FP16Quantization,
        "none": NoQuantization
    }
    
    if name not in algorithms:
        raise ValueError(f"Unknown quantization algorithm: {name}. Available: {list(algorithms.keys())}")
    
    return algorithms[name](**kwargs)

# === Configuration ===

class CompressionConfig:
    """Configuration for tensor compression."""
    
    def __init__(self,
                 compression: str = "none",
                 quantization: str = "none",
                 compression_kwargs: Optional[Dict] = None,
                 quantization_kwargs: Optional[Dict] = None):
        """Initialize compression configuration."""
        self.compression = compression
        self.quantization = quantization
        self.compression_kwargs = compression_kwargs or {}
        self.quantization_kwargs = quantization_kwargs or {}
    
    def create_tensor_compression(self) -> TensorCompression:
        """Create TensorCompression instance from config."""
        comp_algo = create_compression_algorithm(self.compression, **self.compression_kwargs)
        quant_algo = create_quantization_algorithm(self.quantization, **self.quantization_kwargs)
        return TensorCompression(comp_algo, quant_algo)
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'CompressionConfig':
        """Create config from dictionary."""
        return cls(
            compression=config.get("compression", "none"),
            quantization=config.get("quantization", "none"),
            compression_kwargs=config.get("compression_kwargs", {}),
            quantization_kwargs=config.get("quantization_kwargs", {})
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "compression": self.compression,
            "quantization": self.quantization,
            "compression_kwargs": self.compression_kwargs,
            "quantization_kwargs": self.quantization_kwargs
        }

# === Presets ===

COMPRESSION_PRESETS = {
    "none": CompressionConfig("none", "none"),
    "fast": CompressionConfig("lz4", "none", {"compression_level": 1}),
    "balanced": CompressionConfig("gzip", "fp16", {"compression_level": 6}),
    "maximum": CompressionConfig("gzip", "int8", {"compression_level": 9}),
    "fp16_only": CompressionConfig("none", "fp16"),
    "int8_only": CompressionConfig("none", "int8"),
}

def get_compression_preset(name: str) -> CompressionConfig:
    """Get predefined compression preset."""
    if name not in COMPRESSION_PRESETS:
        raise ValueError(f"Unknown preset: {name}. Available: {list(COMPRESSION_PRESETS.keys())}")
    return COMPRESSION_PRESETS[name]