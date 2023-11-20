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

def mean_images(ims):
    return np_to_pil(np.mean(np.stack([pil_to_np(im) if isinstance(im, Image.Image) else im for im in ims]), axis = 0))

def np_to_pil(npim):
    return Image.fromarray(skimage.util.img_as_ubyte(npim))

def pil_to_np(im):
    return skimage.util.img_as_float(im)

def save_image(im, path, archive_mode = False):
    tmp_path = path.with_suffix(".tmp")
    kwargs = dict(
        optimize = True,
        compress_level = 9,
    ) if archive_mode else {}
    im.save(tmp_path, "PNG", **kwargs)
    tmp_path.rename(path)
