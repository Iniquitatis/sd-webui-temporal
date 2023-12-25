import numpy as np
from scipy import stats

def average_array(arr, algo, axis, weights = None):
    if algo == "harmonic_mean":
        return stats.hmean(arr + 1.0, axis = axis, weights = match_array_dimensions(weights, arr, axis) if weights is not None else None) - 1.0
    elif algo == "geometric_mean":
        return stats.gmean(arr + 1.0, axis = axis, weights = match_array_dimensions(weights, arr, axis) if weights is not None else None) - 1.0
    elif algo == "arithmetic_mean":
        if weights is not None:
            return np.average(arr, axis, weights)
        else:
            return np.mean(arr, axis)
    elif algo == "root_mean_square":
        return np.sqrt(np.average(np.square(arr + 1.0), axis, weights)) - 1.0
    elif algo == "median":
        return np.median(arr, axis)
    else:
        raise Exception(f"Unknown algorithm name: {algo}")

def make_eased_weight_array(count, easing):
    return (np.linspace(1, count, count, dtype = np.float_) / count) ** easing

def match_array_dimensions(arr, ref, axis):
    return np.reshape(arr, (1,) * len(ref.shape[:axis]) + arr.shape + (1,) * len(ref.shape[axis + arr.ndim:]))

def load_array(path):
    return np.load(path)["arr_0"]

def save_array(arr, path):
    np.savez_compressed(path, arr)
