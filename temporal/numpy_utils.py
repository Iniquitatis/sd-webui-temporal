import numpy as np
from scipy import stats

def average_array(arr, axis, trim = 0.0, power = 1.0, weights = None):
    if trim == 0.5:
        return np.median(arr, axis)
    elif trim > 0.0:
        arr = stats.trimboth(arr, trim, axis)
        weights = None

    if weights is not None:
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

def make_eased_weight_array(count, easing):
    return (np.linspace(1, count, count, dtype = np.float_) / count) ** easing

def match_array_dimensions(arr, ref, axis):
    return np.reshape(arr, (1,) * len(ref.shape[:axis]) + arr.shape + (1,) * len(ref.shape[axis + arr.ndim:]))

def load_array(path):
    return np.load(path)["arr_0"]

def save_array(arr, path):
    np.savez_compressed(path, arr)
