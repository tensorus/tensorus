"""Bounded C-order NPY decoding for resumable upload finalization."""

import hashlib
import io
import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import IO, TYPE_CHECKING, Any

import numpy as np

from tensorus.arrays import BLOCK_ELEMENTS, Array, canonical_json
from tensorus.errors import LimitExceeded, StorageError, TensorusError
from tensorus.serialization import read_npy_header

if TYPE_CHECKING:
    from _hashlib import HASH


@dataclass(frozen=True)
class StreamedNpy:
    stream: IO[bytes]
    shape: tuple[int, ...]
    dtype: np.dtype[Any]
    offset: int

    @classmethod
    def open(cls, stream: IO[bytes], size: int, max_bytes: int) -> "StreamedNpy":
        stream.seek(0)
        shape, fortran_order, dtype = read_npy_header(stream, size, max_bytes)
        if fortran_order:
            raise TensorusError("Larger resumable NPY uploads require C-order input")
        if math.prod(max(1, dimension) for dimension in shape) > np.iinfo(np.intp).max // (
            dtype.itemsize
        ):
            raise LimitExceeded("NPY dimensions exceed the platform index limit")
        return cls(stream=stream, shape=shape, dtype=dtype, offset=stream.tell())

    @property
    def count(self) -> int:
        return math.prod(self.shape)

    @property
    def nbytes(self) -> int:
        return self.count * self.dtype.itemsize

    def digest(self) -> "HASH":
        result = hashlib.sha256()
        result.update(
            canonical_json({"dtype": self.dtype.name, "shape": list(self.shape)}).encode()
        )
        result.update(b"\0")
        return result

    def read_values(self, count: int, raw_digest: "HASH | None" = None) -> Array:
        size = count * self.dtype.itemsize
        payload = self.stream.read(size)
        if len(payload) != size:
            raise StorageError("Staged NPY payload was truncated during finalization")
        if raw_digest is not None:
            raw_digest.update(payload)
        values = np.frombuffer(payload, dtype=self.dtype)
        if not np.isfinite(values).all():
            raise TensorusError(
                "NaN and infinity are not accepted; clean missing values explicitly"
            )
        return np.array(values, dtype=self.dtype.newbyteorder("<"), order="C", copy=True)

    def blocks(
        self, checkpoint: Callable[[], None], raw_digest: "HASH | None" = None
    ) -> Iterator[Array]:
        self.stream.seek(self.offset)
        remaining = self.count
        while remaining:
            checkpoint()
            count = min(BLOCK_ELEMENTS, remaining)
            yield self.read_values(count, raw_digest)
            remaining -= count

    def checked_blocks(self, checksum: str, checkpoint: Callable[[], None]) -> Iterator[Array]:
        digest = self.digest()
        for block in self.blocks(checkpoint):
            digest.update(memoryview(block).cast("B"))
            yield block
        if digest.hexdigest() != checksum:
            raise StorageError("Staged NPY values changed during finalization")

    def scan(
        self, checkpoint: Callable[[], None], expected_file_checksum: str | None = None
    ) -> tuple[str, dict[str, object]]:
        digest = self.digest()
        raw_digest = hashlib.sha256()
        self.stream.seek(0)
        header = self.stream.read(self.offset)
        if len(header) != self.offset:
            raise StorageError("Staged NPY header was truncated during finalization")
        header_stream = io.BytesIO(header)
        try:
            shape, fortran_order, dtype = read_npy_header(
                header_stream, self.offset + self.nbytes, self.nbytes
            )
        except TensorusError as exc:
            raise StorageError("Staged NPY header changed during finalization") from exc
        if (
            shape != self.shape
            or fortran_order
            or dtype != self.dtype
            or header_stream.tell() != self.offset
        ):
            raise StorageError("Staged NPY header changed during finalization")
        raw_digest.update(header)
        minimum: Any = None
        maximum: Any = None
        for block in self.blocks(checkpoint, raw_digest):
            digest.update(memoryview(block).cast("B"))
            block_minimum, block_maximum = block.min(), block.max()
            minimum = block_minimum if minimum is None else np.minimum(minimum, block_minimum)
            maximum = block_maximum if maximum is None else np.maximum(maximum, block_maximum)
        if self.stream.read(1):
            raise StorageError("Staged NPY payload has unexpected trailing bytes")
        if expected_file_checksum is not None and raw_digest.hexdigest() != expected_file_checksum:
            raise StorageError("Staged NPY file does not match the verified upload checksum")
        checksum = digest.hexdigest()
        result: dict[str, object] = {"count": self.count, "overflow": []}
        if not self.count:
            result.update({"min": None, "max": None, "mean": None, "std": None, "norm": 0.0})
            return checksum, result
        scale = max(abs(np.longdouble(minimum)), abs(np.longdouble(maximum)))
        with np.errstate(over="ignore", invalid="ignore"):
            total, squares = np.longdouble(0), np.longdouble(0)
            if scale:
                for block in self.checked_blocks(checksum, checkpoint):
                    scaled = block.astype(np.longdouble) / scale
                    total += np.sum(scaled)
                    squares += np.sum(scaled * scaled)
            mean = total / self.count
            variance = np.longdouble(0)
            if scale:
                for block in self.checked_blocks(checksum, checkpoint):
                    centered = block.astype(np.longdouble) / scale - mean
                    variance += np.sum(centered * centered)
            raw = {
                "min": np.longdouble(minimum),
                "max": np.longdouble(maximum),
                "mean": scale * mean,
                "std": scale * np.sqrt(variance / self.count),
                "norm": scale * np.sqrt(squares),
            }
        overflow: list[str] = []
        for key, value in raw.items():
            converted = float(value)
            if math.isfinite(converted):
                result[key] = converted
            else:
                result[key] = None
                overflow.append(key)
        result["overflow"] = overflow
        return checksum, result
