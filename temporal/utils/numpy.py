from io import BytesIO
from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from temporal.utils.base64 import decode, decode_with_mime_type, encode, encode_with_mime_type


IntType = np.int32
IntArray = NDArray[IntType]
FloatType = np.float64
FloatArray = NDArray[FloatType]


def array_to_base64(arr: FloatArray, with_mime_type: bool = True) -> str:
    with BytesIO() as stream:
        np.savez_compressed(stream, arr)

        if with_mime_type:
            return encode_with_mime_type("application", "octet-stream", stream.getvalue())
        else:
            return encode(stream.getvalue())


def average_array(arr: FloatArray, axis: int, trim: float = 0.0, power: float = 1.0, weights: Optional[FloatArray] = None) -> FloatArray:
    if trim == 0.5:
        return np.median(arr, axis)
    elif trim > 0.0:
        arr = stats.trimboth(arr, trim, axis)
        weights = None

    # NOTE: That power check is needed for `np.average`, but not for `stats.x`
    if weights is not None and power not in (1.0, 2.0, 3.0):
        weights = match_array_dimensions(weights, arr, axis)

    if power != 1.0:
        arr = arr + 1.0

    if power == -1.0:
        result = stats.hmean(arr, axis = axis, weights = weights)
    elif power == 0.0:
        result = stats.gmean(arr, axis = axis, weights = weights)
    elif power == 1.0:
        result = np.average(arr, axis, weights)
    elif power == 2.0:
        result = np.sqrt(np.average(np.square(arr), axis, weights))
    elif power == 3.0:
        result = np.cbrt(np.average(np.power(arr, 3.0), axis, weights))
    else:
        result = stats.pmean(arr, power, axis = axis, weights = weights)

    if power != 1.0:
        result -= 1.0

    return result


def base64_to_array(text: str, with_mime_type: bool = True) -> FloatArray:
    if with_mime_type:
        type, subtype, data = decode_with_mime_type(text)
    else:
        type, subtype, data = "application", "octet-stream", decode(text)

    if type != "application" or subtype != "octet-stream":
        raise ValueError

    return np.load(BytesIO(data))["arr_0"]


def make_eased_weight_array(count: int, easing: float) -> FloatArray:
    return (np.linspace(1, count, count, dtype = FloatType) / count) ** easing


def match_array_dimensions(arr: FloatArray, ref: FloatArray, axis: int) -> FloatArray:
    return np.reshape(arr, (1,) * len(ref.shape[:axis]) + arr.shape + (1,) * len(ref.shape[axis + arr.ndim:]))


def load_array(path: str | Path) -> FloatArray:
    return np.load(path)["arr_0"]


def random_array(shape: tuple[int, ...], low: float = 0.0, high: float = 1.0, seed: int = -1) -> FloatArray:
    return np.divide(np.random.default_rng(seed).integers(
        low = round(low * 1e9),
        high = round(high * 1e9),
        size = shape,
        dtype = IntType,
        endpoint = True,
    ), 1e9, dtype = FloatType)


def saturate_array(arr: FloatArray) -> FloatArray:
    return np.clip(arr, 0.0, 1.0)


def save_array(arr: FloatArray, path: str | Path) -> None:
    np.savez_compressed(path, arr)


def stretch_array(arr: FloatArray, new_length: int) -> FloatArray:
    return np.interp(np.arange(new_length), np.linspace(0, new_length - 1, len(arr)), arr)
