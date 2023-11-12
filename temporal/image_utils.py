import numpy as np
from PIL import Image

def generate_noise_image(size, seed):
    return Image.fromarray(np.random.default_rng(seed).integers(0, 256, size = (size[1], size[0], 3), dtype = "uint8"))

def load_image(path):
    im = Image.open(path)
    im.load()
    return im

def match_image(im, reference, mode = True, size = True):
    if mode and im.mode != reference.mode:
        im = im.convert(reference.mode)

    if size and im.size != reference.size:
        im = im.resize(reference.size, Image.Resampling.LANCZOS)

    return im
