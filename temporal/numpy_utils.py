import numpy as np

def load_array(path):
    with open(path, "rb") as file:
        return np.load(file)

def save_array(arr, path):
    with open(path, "wb") as file:
        np.save(file, arr)
