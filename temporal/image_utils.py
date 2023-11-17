import numpy as np
import skimage
from PIL import Image

def generate_noise_image(size, seed):
    return Image.fromarray(np.random.default_rng(seed).integers(0, 256, size = (size[1], size[0], 3), dtype = "uint8"))

def load_image(path):
    im = Image.open(path)
    im.load()
    return im

def match_image(im, reference, mode = True, size = True):
    if is_np := isinstance(im, np.ndarray):
        im = Image.fromarray(skimage.util.img_as_ubyte(im))

    if isinstance(reference, np.ndarray):
        reference = Image.fromarray(skimage.util.img_as_ubyte(reference))

    if mode and im.mode != reference.mode:
        im = im.convert(reference.mode)

    if size and im.size != reference.size:
        im = im.resize(reference.size, Image.Resampling.LANCZOS)

    return skimage.util.img_as_float(im) if is_np else im

def np_to_pil(npim):
    return Image.fromarray(skimage.util.img_as_ubyte(npim))

def pil_to_np(im):
    return skimage.util.img_as_float(im)
