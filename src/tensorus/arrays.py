import hashlib
import json
import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any, Literal, NoReturn

import numpy as np
from numpy.typing import NDArray

from tensorus.errors import LimitExceeded, StorageError, TensorusError
from tensorus.models import DType, SimilarityEngine

Array = NDArray[Any]
BLOCK_ELEMENTS = max(1, 1024**2 // np.dtype(np.longdouble).itemsize)
CHUNKED_BLOCK_BYTES = 4 * 1024**2
CHUNKED_MAX_BLOCKS = 8192
CHUNKED_DISPERSION_PASSES = 3
CHUNKED_DISPERSION_WORK_ARRAYS = 20
CHUNKED_DISPERSION_ITEMSIZE = np.dtype("float64").itemsize


def decode_unique_json(data: bytes) -> Any:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON object key")
            result[key] = value
        return result

    def invalid_constant(_value: str) -> NoReturn:
        raise ValueError("Nonfinite JSON number")

    return json.loads(
        data.decode("utf-8"), object_pairs_hook=unique_object, parse_constant=invalid_constant
    )


def canonical_json(value: object) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (ValueError, TypeError, RecursionError) as exc:
        raise TensorusError("Expected finite, JSON-serializable values") from exc


def canonical_hash(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def _check_cpu_input(data: object) -> None:
    if np.ma.isMaskedArray(data):
        raise TensorusError("Masked arrays require explicit data and validity-mask versions")
    if hasattr(data, "__cuda_array_interface__"):
        raise TensorusError("GPU arrays are not supported; transfer values to CPU explicitly")
    device_method = getattr(data, "__dlpack_device__", None)
    if callable(device_method):
        try:
            device = device_method()
        except (BufferError, RuntimeError, TypeError, ValueError) as exc:
            raise TensorusError("Unable to determine the array device") from exc
        if not isinstance(device, tuple) or len(device) != 2 or device[0] != 1:
            raise TensorusError("GPU arrays are not supported; transfer values to CPU explicitly")


def _check_array_input(
    data: object,
) -> tuple[bool, int | float | None, int | float | None, int | None, int | None]:
    if isinstance(data, (list, tuple)) and type(data) not in (list, tuple):
        raise TensorusError("List and tuple subclasses are not supported as tensor containers")
    if not isinstance(data, (list, tuple)):
        _check_cpu_input(data)
        return False, None, None, None, None
    stack: list[Iterator[Any]] = [iter(data)]
    literals = True
    lower: int | float | None = None
    upper: int | float | None = None
    integer_lower: int | None = None
    integer_upper: int | None = None
    while stack:
        try:
            value = next(stack[-1])
        except StopIteration:
            stack.pop()
            continue
        if isinstance(value, (list, tuple)) and type(value) not in (list, tuple):
            raise TensorusError("List and tuple subclasses are not supported as tensor containers")
        if type(value) in (list, tuple):
            if len(stack) >= 16:
                raise LimitExceeded("Tensors may have at most 16 dimensions")
            stack.append(iter(value))
        elif type(value) in (bool, int, float) or isinstance(
            value, (np.bool_, np.integer, np.floating)
        ):
            numeric: int | float
            if isinstance(value, np.integer):
                numeric = int(value)
            elif isinstance(value, np.floating):
                numeric = float(value)
            elif isinstance(value, np.bool_):
                numeric = bool(value)
            else:
                numeric = value
            lower = numeric if lower is None or numeric < lower else lower
            upper = numeric if upper is None or numeric > upper else upper
            if type(numeric) is int:
                integer_lower = (
                    numeric if integer_lower is None or numeric < integer_lower else integer_lower
                )
                integer_upper = (
                    numeric if integer_upper is None or numeric > integer_upper else integer_upper
                )
        else:
            literals = False
            _check_cpu_input(value)
            if isinstance(value, np.ndarray) and value.dtype.kind in "iu" and value.size:
                minimum, maximum = int(value.min()), int(value.max())
                integer_lower = minimum if integer_lower is None else min(integer_lower, minimum)
                integer_upper = maximum if integer_upper is None else max(integer_upper, maximum)
    return literals, lower, upper, integer_lower, integer_upper


def _check_inferred_integer_precision(
    source: Array, integer_lower: int | None, integer_upper: int | None
) -> None:
    if source.dtype.kind != "f" or integer_lower is None or integer_upper is None:
        return
    limit = 2 ** (np.finfo(source.dtype).nmant + 1)
    if integer_lower < -limit or integer_upper > limit:
        raise TensorusError(
            "An implicit float inference may round integer inputs; "
            "provide an explicitly typed array or dtype"
        )


def validated_array(data: object, max_bytes: int, dtype: str | DType | None = None) -> Array:
    literals, lower, upper, integer_lower, integer_upper = _check_array_input(data)
    try:
        source = np.asarray(data)
    except (ValueError, TypeError, OverflowError) as exc:
        raise TensorusError("Expected a rectangular array of supported real numbers") from exc
    if source.dtype.name not in {str(item) for item in DType}:
        raise TensorusError(f"Unsupported source dtype: {source.dtype}")
    if source.ndim > 16:
        raise LimitExceeded("Tensors may have at most 16 dimensions")
    if source.nbytes > max_bytes:
        raise LimitExceeded(f"Tensor source exceeds the {max_bytes}-byte limit")
    if not np.isfinite(source).all():
        raise TensorusError("NaN and infinity are not accepted; clean missing values explicitly")
    array = source
    if dtype is None:
        _check_inferred_integer_precision(source, integer_lower, integer_upper)
    else:
        try:
            target = np.dtype(str(dtype))
        except (ValueError, TypeError) as exc:
            raise TensorusError("Unsupported dtype for tensor conversion") from exc
        if target.name not in {str(item) for item in DType}:
            raise TensorusError(f"Unsupported dtype: {target}")
        if target.kind in "iu" and not literals:
            _check_inferred_integer_precision(source, integer_lower, integer_upper)
        if target.kind in "iu" and source.size:
            if not literals:
                lower, upper = source.min().item(), source.max().item()
            limits = np.iinfo(target)
            if lower is None or upper is None or lower < limits.min or upper > limits.max:
                raise TensorusError("Conversion would exceed the target integer range")
        try:
            # Literal sequences preserve wide integer values through explicit conversion.
            with np.errstate(over="raise", invalid="raise"):
                array = np.asarray(data if literals else source, dtype=target)
        except (ValueError, TypeError, OverflowError, FloatingPointError) as exc:
            raise TensorusError("Tensor dtype conversion failed or overflowed") from exc
    if array.nbytes > max_bytes:
        raise LimitExceeded(f"Tensor exceeds the {max_bytes}-byte limit")
    if not np.isfinite(array).all():
        raise TensorusError("NaN and infinity are not accepted; clean missing values explicitly")
    return np.array(array, dtype=array.dtype.newbyteorder("<"), order="C", copy=True)


def array_checksum(array: Array) -> str:
    digest = hashlib.sha256()
    digest.update(canonical_json({"dtype": array.dtype.name, "shape": list(array.shape)}).encode())
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def descriptor_nbytes(record: Mapping[str, Any]) -> int:
    shape, dtype, nbytes = record.get("shape"), record.get("dtype"), record.get("nbytes")
    if (
        not isinstance(shape, list)
        or len(shape) > 16
        or any(type(dimension) is not int or dimension < 0 for dimension in shape)
        or not isinstance(dtype, str)
        or dtype not in DType
        or type(nbytes) is not int
    ):
        raise StorageError("Stored tensor descriptor is invalid")
    expected = math.prod(shape) * np.dtype(dtype).itemsize
    if expected != nbytes:
        raise StorageError("Stored tensor byte count does not match its descriptor")
    return expected


def reduced_shape(shape: tuple[int, ...], axes: tuple[int, ...], keepdims: bool) -> tuple[int, ...]:
    return tuple(
        1 if axis in axes else dimension
        for axis, dimension in enumerate(shape)
        if keepdims or axis not in axes
    )


def chunked_block_layout(shape: tuple[int, ...], itemsize: int) -> tuple[int | None, int, int]:
    if len(shape) > 16 or any(type(dimension) is not int or dimension < 0 for dimension in shape):
        raise StorageError("Stored tensor shape is invalid")
    if (
        itemsize < 1
        or itemsize > CHUNKED_BLOCK_BYTES
        or math.prod(max(1, dimension) for dimension in shape) > np.iinfo(np.intp).max // itemsize
    ):
        raise LimitExceeded("Tensor dimensions exceed the chunked index limit")
    if not shape:
        return None, 0, 1
    if not math.prod(shape):
        return None, 0, 0
    limit = CHUNKED_BLOCK_BYTES // itemsize
    suffix = 1
    for axis in range(len(shape) - 1, -1, -1):
        if suffix * shape[axis] > limit:
            width = limit // suffix
            count = math.prod(shape[:axis]) * ((shape[axis] + width - 1) // width)
            return axis, width, count
        suffix *= shape[axis]
    return None, 0, 1


def chunked_c_order_selections(
    shape: tuple[int, ...],
    chunks: tuple[int, ...],
    itemsize: int,
    *,
    align_chunks: bool = True,
) -> Iterator[tuple[slice, ...]]:
    axis, width, count = chunked_block_layout(shape, itemsize)
    if len(chunks) != len(shape) or any(chunk < 1 for chunk in chunks):
        raise StorageError("Object has an invalid chunk grid")
    if count > CHUNKED_MAX_BLOCKS:
        raise LimitExceeded("Tensor exceeds the chunked block-count limit")
    if not count:
        return
    if axis is None:
        yield (slice(None),) * len(shape)
        return
    for prefix in np.ndindex(shape[:axis]):
        start = 0
        while start < shape[axis]:
            chunk = chunks[axis]
            if not align_chunks:
                stop = min(start + width, shape[axis])
            elif width >= chunk:
                stop = min(start + (width // chunk) * chunk, shape[axis])
            else:
                stop = min(start + width, ((start // chunk) + 1) * chunk, shape[axis])
            yield (
                *(slice(index, index + 1) for index in prefix),
                slice(start, stop),
                *(slice(None) for _ in shape[axis + 1 :]),
            )
            start = stop


def require_float64_integer_precision(array: Array) -> None:
    if np.issubdtype(array.dtype, np.integer) and array.size:
        limits = np.iinfo(array.dtype)
        if (limits.max > 2**53 and int(array.max()) > 2**53) or (
            limits.min < -(2**53) and int(array.min()) < -(2**53)
        ):
            raise TensorusError(
                "Arithmetic promotes to float64; integers beyond 2**53 require an explicit cast"
            )


def accumulate_chunked_sum(
    output: Array,
    block: Array,
    selection: tuple[slice, ...],
    axes: tuple[int, ...],
    keepdims: bool,
    mask: Array | None = None,
) -> None:
    target = chunked_reduction_target(selection, axes, keepdims)
    contiguous = np.asarray(block, order="C")
    if mask is None:
        subtotal = (
            np.sum(contiguous, axis=axes, keepdims=True, dtype=np.float64)
            if keepdims
            else np.sum(contiguous, axis=axes, keepdims=False, dtype=np.float64)
        )
    else:
        subtotal = (
            np.sum(contiguous, axis=axes, keepdims=True, dtype=np.float64, where=mask)
            if keepdims
            else np.sum(contiguous, axis=axes, keepdims=False, dtype=np.float64, where=mask)
        )
    output[target] += subtotal


def chunked_reduction_target(
    selection: tuple[slice, ...], axes: tuple[int, ...], keepdims: bool
) -> tuple[slice, ...]:
    return tuple(
        slice(None) if axis in axes else item
        for axis, item in enumerate(selection)
        if keepdims or axis not in axes
    )


def extrema_identity(dtype: np.dtype[Any], minimum: bool) -> bool | int | float:
    if dtype.kind == "b":
        return minimum
    if dtype.kind in "iu":
        bounds = np.iinfo(dtype)
        return int(bounds.max if minimum else bounds.min)
    return np.inf if minimum else -np.inf


def accumulate_chunked_extrema(
    output: Array,
    block: Array,
    selection: tuple[slice, ...],
    axes: tuple[int, ...],
    keepdims: bool,
    minimum: bool,
    mask: Array | None = None,
    included: Array | None = None,
) -> None:
    target = chunked_reduction_target(selection, axes, keepdims)
    reduce_fn: Callable[..., Any] = np.min if minimum else np.max
    combine_fn = np.minimum if minimum else np.maximum
    initial = extrema_identity(block.dtype, minimum)
    if mask is None:
        subtotal = reduce_fn(block, axis=axes, keepdims=keepdims, initial=initial)
    else:
        subtotal = reduce_fn(block, axis=axes, keepdims=keepdims, where=mask, initial=initial)
        if included is None:
            raise ValueError("A masked extremum requires an inclusion accumulator")
        subtotal_included = (
            np.any(mask, axis=axes, keepdims=True)
            if keepdims
            else np.any(mask, axis=axes, keepdims=False)
        )
        included[target] |= subtotal_included
    output[target] = combine_fn(output[target], subtotal)


def accumulate_chunked_any(
    output: Array,
    block: Array,
    selection: tuple[slice, ...],
    axes: tuple[int, ...],
    keepdims: bool,
) -> None:
    target = chunked_reduction_target(selection, axes, keepdims)
    subtotal = (
        np.any(block, axis=axes, keepdims=True)
        if keepdims
        else np.any(block, axis=axes, keepdims=False)
    )
    output[target] |= subtotal


def accumulate_chunked_included_count(
    output: Array,
    mask: Array,
    selection: tuple[slice, ...],
    axes: tuple[int, ...],
    keepdims: bool,
) -> None:
    target = chunked_reduction_target(selection, axes, keepdims)
    output[target] += np.count_nonzero(mask, axis=axes, keepdims=keepdims)


def mean_from_chunked_sums(sums: Array, counts: Array | int) -> Array:
    result = np.zeros_like(sums)
    with np.errstate(under="ignore"):
        np.divide(sums, counts, out=result, where=np.asarray(counts) != 0)
    if np.any((sums != 0) & (result == 0)):
        raise TensorusError("Chunked mean underflowed float64 to zero")
    return result


def chunked_dispersion(
    shape: tuple[int, ...],
    axes: tuple[int, ...],
    keepdims: bool,
    blocks: Callable[[], Iterator[tuple[tuple[slice, ...], list[Array]]]],
    *,
    masked: bool,
    variance: bool,
    max_working_bytes: int,
) -> Array:
    output_shape = reduced_shape(shape, axes, keepdims)
    if (
        math.prod(output_shape) * CHUNKED_DISPERSION_ITEMSIZE * CHUNKED_DISPERSION_WORK_ARRAYS
        > max_working_bytes
    ):
        raise LimitExceeded("Chunked dispersion exceeds the output working-memory limit")
    low = np.full(output_shape, np.inf, dtype=np.float64)
    high = np.full(output_shape, -np.inf, dtype=np.float64)
    counts = np.zeros(output_shape, dtype=np.int64)

    def scan(
        accumulate: Callable[[tuple[slice, ...], list[Array]], None],
    ) -> None:
        failure: TensorusError | None = None
        for selection, values in blocks():
            if failure is not None:
                continue
            try:
                with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
                    accumulate(selection, values)
            except TensorusError as exc:
                failure = exc
            except (ValueError, FloatingPointError, OverflowError) as exc:
                failure = TensorusError(f"Chunked dispersion failed numerically: {exc}")
        if failure is not None:
            raise failure

    def bounds(selection: tuple[slice, ...], values: list[Array]) -> None:
        block = values[0]
        mask = values[1] if masked else None
        if not np.isfinite(block).all():
            raise TensorusError("Chunked dispersion requires finite input values")
        require_float64_integer_precision(block[mask] if mask is not None else block)
        numeric = block.astype(np.float64, copy=False)
        target = chunked_reduction_target(selection, axes, keepdims)
        where = mask if mask is not None else True
        minimum_fn: Callable[..., Any] = np.min
        maximum_fn: Callable[..., Any] = np.max
        minimum = minimum_fn(numeric, axis=axes, keepdims=keepdims, where=where, initial=np.inf)
        maximum = maximum_fn(numeric, axis=axes, keepdims=keepdims, where=where, initial=-np.inf)
        low[target] = np.minimum(low[target], minimum)
        high[target] = np.maximum(high[target], maximum)
        counts[target] += (
            np.count_nonzero(mask, axis=axes, keepdims=keepdims)
            if mask is not None
            else math.prod(block.shape[axis] for axis in axes)
        )

    scan(bounds)
    included = counts != 0
    np.copyto(low, 0, where=~included)
    np.copyto(high, 0, where=~included)
    midpoint = low / 2 + high / 2
    scale = np.maximum(np.abs(low - midpoint), np.abs(high - midpoint))
    normalized_sum = np.zeros(output_shape, dtype=np.float64)

    def expanded(value: Array, selection: tuple[slice, ...], block: Array) -> Array:
        target = chunked_reduction_target(selection, axes, keepdims)
        dimensions = tuple(1 if index in axes else size for index, size in enumerate(block.shape))
        return np.asarray(value[target]).reshape(dimensions)

    def normalized(
        selection: tuple[slice, ...], values: list[Array]
    ) -> tuple[Array, tuple[slice, ...], Array | None]:
        block = values[0]
        mask = values[1] if masked else None
        where = mask if mask is not None else True
        deviations = np.zeros(block.shape, dtype=np.float64)
        np.subtract(block, expanded(midpoint, selection, block), out=deviations, where=where)
        denominator = expanded(scale, selection, block)
        np.divide(deviations, denominator, out=deviations, where=denominator != 0)
        return deviations, chunked_reduction_target(selection, axes, keepdims), mask

    def accumulate_mean(selection: tuple[slice, ...], values: list[Array]) -> None:
        deviations, target, _ = normalized(selection, values)
        subtotal = (
            np.sum(deviations, axis=axes, keepdims=True, dtype=np.float64)
            if keepdims
            else np.sum(deviations, axis=axes, keepdims=False, dtype=np.float64)
        )
        normalized_sum[target] += subtotal

    scan(accumulate_mean)
    np.divide(normalized_sum, counts, out=normalized_sum, where=included)
    normalized_low = np.divide(low - midpoint, scale, out=np.zeros_like(low), where=scale != 0)
    normalized_high = np.divide(high - midpoint, scale, out=np.zeros_like(high), where=scale != 0)
    centered_scale = np.maximum(
        np.abs(normalized_low - normalized_sum), np.abs(normalized_high - normalized_sum)
    )
    squares = np.zeros(output_shape, dtype=np.float64)

    def accumulate_squares(selection: tuple[slice, ...], values: list[Array]) -> None:
        deviations, target, mask = normalized(selection, values)
        where = mask if mask is not None else True
        np.subtract(
            deviations,
            expanded(normalized_sum, selection, values[0]),
            out=deviations,
            where=where,
        )
        denominator = expanded(centered_scale, selection, values[0])
        np.divide(deviations, denominator, out=deviations, where=denominator != 0)
        np.square(deviations, out=deviations)
        subtotal = (
            np.sum(deviations, axis=axes, keepdims=True, dtype=np.float64)
            if keepdims
            else np.sum(deviations, axis=axes, keepdims=False, dtype=np.float64)
        )
        squares[target] += subtotal

    scan(accumulate_squares)
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        try:
            np.divide(squares, counts, out=squares, where=included)
            spread = scale * (centered_scale * np.sqrt(squares))
            if np.any((low != high) & (spread == 0)):
                raise TensorusError("Standard deviation underflowed float64")
            if variance:
                result = np.square(spread)
                if np.any((spread != 0) & (result == 0)):
                    raise TensorusError("Variance underflowed float64")
                return result
            return spread
        except FloatingPointError as exc:
            raise TensorusError(f"Chunked dispersion failed numerically: {exc}") from exc


def positional_selection(
    shape: Sequence[int], requested: Sequence[Sequence[int | None]]
) -> tuple[tuple[slice, ...], tuple[int, ...]]:
    if len(requested) > len(shape):
        raise TensorusError("Selection has more dimensions than the tensor")
    selection = tuple(slice(*item) for item in requested)
    selection += (slice(None),) * (len(shape) - len(selection))
    try:
        selected_shape = tuple(
            len(range(*item.indices(dimension)))
            for item, dimension in zip(selection, shape, strict=True)
        )
    except OverflowError as exc:
        raise LimitExceeded("Slice dimensions exceed the platform index limit") from exc
    return selection, selected_shape


def _blocks(array: Array) -> Iterator[Array]:
    flat = array.reshape(-1)
    for start in range(0, flat.size, BLOCK_ELEMENTS):
        yield flat[start : start + BLOCK_ELEMENTS].astype(np.longdouble)


def _magnitude(array: Array) -> np.longdouble:
    if not array.size:
        return np.longdouble(0)
    return max(abs(np.longdouble(array.min())), abs(np.longdouble(array.max())))


def statistics(array: Array) -> dict[str, object]:
    result: dict[str, object] = {"count": int(array.size), "overflow": []}
    if not array.size:
        result.update({"min": None, "max": None, "mean": None, "std": None, "norm": 0.0})
        return result
    scale = _magnitude(array)
    with np.errstate(over="ignore", invalid="ignore"):
        total, squares = np.longdouble(0), np.longdouble(0)
        if scale:
            for block in _blocks(array):
                scaled = block / scale
                total += np.sum(scaled)
                squares += np.sum(scaled * scaled)
        mean = total / array.size
        variance = np.longdouble(0)
        if scale:
            for block in _blocks(array):
                centered = block / scale - mean
                variance += np.sum(centered * centered)
        raw = {
            "min": np.longdouble(array.min()),
            "max": np.longdouble(array.max()),
            "mean": scale * mean,
            "std": scale * np.sqrt(variance / array.size),
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
    return result


def extended_precision_bits() -> int:
    return int(np.finfo(np.longdouble).nmant) + 1


def similarity_engine(metric: Literal["cosine", "l2"]) -> SimilarityEngine:
    if metric not in {"cosine", "l2"}:
        raise TensorusError("Unsupported similarity metric")
    return SimilarityEngine(
        algorithm="scaled-longdouble-v1"
        if metric == "cosine"
        else "difference-scaled-longdouble-v2",
        numpy_version=np.__version__,
        longdouble_bits=int(np.finfo(np.longdouble).bits),
        longdouble_mantissa_bits=extended_precision_bits(),
        block_elements=BLOCK_ELEMENTS,
    )


def l2_score(left: Array, right: Array, *, precision: int | None = None) -> float:
    if left.shape != right.shape:
        raise TensorusError("Similarity requires matching shapes")
    precision = extended_precision_bits() if precision is None else precision
    integer_kinds = {"b", "i", "u"}
    left_integer = left.dtype.kind in integer_kinds
    right_integer = right.dtype.kind in integer_kinds
    if left_integer != right_integer:
        integer = left if left_integer else right
        if integer.size and integer.dtype.itemsize * 8 > precision:
            maximum = 2**precision
            if int(integer.min()) < -maximum or int(integer.max()) > maximum:
                raise TensorusError(
                    "Mixed integer/float L2 exceeds this platform's extended-precision range"
                )
    exact_integer_difference = (
        left_integer
        and right_integer
        and max(left.dtype.itemsize, right.dtype.itemsize) * 8 > precision
    )
    first, second = left.reshape(-1), right.reshape(-1)
    scale, squares = np.longdouble(0), np.longdouble(0)
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        for start in range(0, first.size, BLOCK_ELEMENTS):
            a, b = first[start : start + BLOCK_ELEMENTS], second[start : start + BLOCK_ELEMENTS]
            if exact_integer_difference:
                difference = np.asarray(a.astype(object) - b.astype(object), dtype=np.longdouble)
            else:
                difference = a.astype(np.longdouble) - b.astype(np.longdouble)
            if not np.isfinite(difference).all():
                return math.inf
            block_scale = np.max(np.abs(difference), initial=np.longdouble(0))
            if block_scale == 0:
                continue
            if block_scale > scale:
                squares = squares * (scale / block_scale) ** 2 + np.sum(
                    (difference / block_scale) ** 2
                )
                scale = block_scale
            else:
                squares += np.sum((difference / scale) ** 2)
        return float(scale * np.sqrt(squares))


def similarity_score(left: Array, right: Array, metric: Literal["cosine", "l2"]) -> float:
    if left.shape != right.shape or metric not in {"cosine", "l2"}:
        raise TensorusError("Similarity requires matching shapes and a supported metric")
    if metric == "l2":
        return l2_score(left, right)
    left_scale, right_scale = _magnitude(left), _magnitude(right)
    if not left_scale or not right_scale:
        return float(left_scale == right_scale)
    dot, left_squared, right_squared = (np.longdouble(0) for _ in range(3))
    with np.errstate(over="ignore", under="ignore", invalid="raise", divide="raise"):
        for first, second in zip(_blocks(left), _blocks(right), strict=True):
            first, second = first / left_scale, second / right_scale
            dot += np.sum(first * second)
            left_squared += np.sum(first * first)
            right_squared += np.sum(second * second)
        return float(np.clip(dot / (np.sqrt(left_squared) * np.sqrt(right_squared)), -1, 1))


def chunk_shape(
    shape: tuple[int, ...], itemsize: int, target_bytes: int = 1024**2
) -> tuple[int, ...]:
    if not shape:
        return ()
    chunks = [max(1, dimension) for dimension in shape]
    while math.prod(chunks) * itemsize > target_bytes:
        largest = max(range(len(chunks)), key=chunks.__getitem__)
        chunks[largest] = (chunks[largest] + 1) // 2
    return tuple(chunks)


Record = dict[str, Any]
MAX_NPY_TILE_BYTES = 1024**2
MAX_SNAPSHOT_TENSORS = 10_000
MAX_TENANT_STAGING_WORK_BYTES = 2**50
MAX_TENANT_STAGING_INPUT_BYTES = 2**50


def version_integrity_descriptor(record: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "id",
        "tenant_id",
        "tensor_id",
        "dataset_id",
        "version",
        "name",
        "dataset",
        "created_by",
        "created_at",
        "checksum",
        "shape",
        "dtype",
        "nbytes",
        "metadata",
        "operation",
        "statistics",
        "parents",
        "write_contract",
    )
    return {**{field: record[field] for field in fields}, "access": record.get("access")}


def ingestion_request_hash(
    dataset: str,
    name: str,
    checksum: str,
    metadata: Mapping[str, Any],
    expected_version: int | None,
    parents: list[str] | None = None,
    operation: dict[str, Any] | None = None,
) -> str:
    return hashlib.sha256(
        canonical_json(
            {
                "dataset": dataset,
                "name": name,
                "checksum": checksum,
                "metadata": metadata,
                "expected_version": expected_version,
                "parents": parents or [],
                "operation": operation,
            }
        ).encode()
    ).hexdigest()
