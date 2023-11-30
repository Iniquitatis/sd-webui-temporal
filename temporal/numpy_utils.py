import numpy as np

def load_array(path):
    return np.load(path)["arr_0"]

def save_array(arr, path):
    np.savez_compressed(path, arr)
