import io
import math
from typing import IO, Any

import numpy as np

from tensorus.arrays import Array, validated_array
from tensorus.errors import LimitExceeded, TensorusError
from tensorus.models import NPY_MAX_OVERHEAD_BYTES as NPY_MAX_OVERHEAD_BYTES
from tensorus.models import DType

NPY_MAX_HEADER_BYTES = 16_384


def read_npy_header(
    stream: IO[bytes], payload_size: int, max_bytes: int
) -> tuple[tuple[int, ...], bool, np.dtype[Any]]:
    try:
        version = np.lib.format.read_magic(stream)
        if version not in {(1, 0), (2, 0)}:
            raise TensorusError("Only numeric NPY format versions 1.0 and 2.0 are supported")
        length_bytes = 2 if version == (1, 0) else 4
        encoded_length = stream.read(length_bytes)
        if len(encoded_length) != length_bytes:
            raise TensorusError("Invalid NPY file")
        header_length = int.from_bytes(encoded_length, "little")
        if header_length > NPY_MAX_HEADER_BYTES:
            raise TensorusError("NPY header exceeds the configured byte limit")
        if header_length > payload_size - stream.tell():
            raise TensorusError("NPY payload length does not match its header")
        stream.seek(-length_bytes, io.SEEK_CUR)
        if version == (1, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(
                stream, max_header_size=NPY_MAX_HEADER_BYTES
            )
        else:
            shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(
                stream, max_header_size=NPY_MAX_HEADER_BYTES
            )
        if dtype.hasobject or dtype.name not in {str(item) for item in DType}:
            raise TensorusError("NPY payload must use a supported, non-object numeric dtype")
        if len(shape) > 16 or any(dimension < 0 for dimension in shape):
            raise TensorusError("Invalid NPY shape")
        size = math.prod(shape) * dtype.itemsize
        if size > max_bytes:
            raise LimitExceeded("NPY header declares a tensor beyond the configured byte limit")
        if payload_size - stream.tell() != size:
            raise TensorusError("NPY payload length does not match its header")
        return shape, fortran_order, dtype
    except (ValueError, EOFError, TypeError, OverflowError) as exc:
        raise TensorusError("Invalid NPY file") from exc


def load_npy(payload: bytes, max_bytes: int) -> Array:
    stream = io.BytesIO(payload)
    shape, fortran_order, dtype = read_npy_header(stream, len(payload), max_bytes)
    try:
        array = np.frombuffer(payload, dtype=dtype, count=math.prod(shape), offset=stream.tell())
        array = array.reshape(shape, order="F" if fortran_order else "C")
        return validated_array(array, max_bytes)
    except (ValueError, EOFError, TypeError, OverflowError) as exc:
        raise TensorusError("Invalid NPY file") from exc


def load_npy_file(stream: IO[bytes], payload_size: int, max_bytes: int) -> Array:
    stream.seek(0)
    shape, fortran_order, dtype = read_npy_header(stream, payload_size, max_bytes)
    try:
        values = np.fromfile(stream, dtype=dtype, count=math.prod(shape))
        if values.size != math.prod(shape):
            raise TensorusError("NPY payload was truncated during decoding")
        return validated_array(
            values.reshape(shape, order="F" if fortran_order else "C"), max_bytes
        )
    except (ValueError, EOFError, TypeError, OverflowError) as exc:
        raise TensorusError("Invalid NPY file") from exc


def dump_npy(array: Array) -> bytes:
    stream = io.BytesIO()
    np.save(stream, array, allow_pickle=False)
    return stream.getvalue()


def dump_npy_header(array: Array) -> bytes:
    stream = io.BytesIO()
    np.lib.format.write_array_header_1_0(stream, np.lib.format.header_data_from_array_1_0(array))
    return stream.getvalue()
