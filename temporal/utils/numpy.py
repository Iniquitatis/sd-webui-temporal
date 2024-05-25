from pathlib import Path
from typing import Optional

import numpy as np
import skimage
from numpy.typing import NDArray
from scipy import stats


FloatArray = NDArray[np.float_]


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


def generate_noise(shape: tuple[int, ...], seed: Optional[int] = None) -> FloatArray:
    return np.random.default_rng(seed).uniform(low = 0.0, high = 1.0 + np.finfo(np.float_).eps, size = shape)


def generate_value_noise(shape: tuple[int, ...], scale: float, octaves: int, lacunarity: float, persistence: float, seed: Optional[int] = None) -> FloatArray:
    noise = generate_noise(shape, seed)

    def scaled_noise(scale: float) -> FloatArray:
        return skimage.transform.warp(noise, skimage.transform.AffineTransform(scale = scale).inverse, order = 3, mode = "symmetric")

    result = np.zeros(shape)
    total_amplitude = 0.0
    amplitude = 0.5

    for _ in range(octaves):
        result += scaled_noise(scale) * amplitude
        total_amplitude += amplitude
        scale /= lacunarity
        amplitude *= persistence

    return result / total_amplitude


def make_eased_weight_array(count: int, easing: float) -> FloatArray:
    return (np.linspace(1, count, count, dtype = np.float_) / count) ** easing


def match_array_dimensions(arr: FloatArray, ref: FloatArray, axis: int) -> FloatArray:
    return np.reshape(arr, (1,) * len(ref.shape[:axis]) + arr.shape + (1,) * len(ref.shape[axis + arr.ndim:]))


def load_array(path: str | Path) -> FloatArray:
    return np.load(path)["arr_0"]


def saturate_array(arr: FloatArray) -> FloatArray:
    return np.clip(arr, 0.0, 1.0)


def save_array(arr: FloatArray, path: str | Path) -> None:
    np.savez_compressed(path, arr)
