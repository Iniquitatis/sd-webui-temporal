import numpy as np

def average_array(arr, algo, axis, weights = None):
    if algo == "mean":
        if weights is not None:
            return np.average(arr, axis, weights)
        else:
            return np.mean(arr, axis)
    elif algo == "median":
        return np.median(arr, axis)
    else:
        raise Exception(f"Unknown algorithm name: {algo}")

def make_eased_weight_array(count, easing):
    return (np.linspace(1, count, count, dtype = np.float_) / count) ** easing

def load_array(path):
    return np.load(path)["arr_0"]

def save_array(arr, path):
    np.savez_compressed(path, arr)
