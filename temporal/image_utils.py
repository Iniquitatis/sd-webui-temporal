import numpy as np
import skimage
from PIL import Image

from temporal.numpy_utils import average_array, generate_noise, generate_value_noise, make_eased_weight_array

def average_images(ims, trimming = 0.0, easing = 0.0, preference = 0.0):
    return ims[0] if len(ims) == 1 else np_to_pil(np.clip(average_array(
        np.stack([pil_to_np(im) if isinstance(im, Image.Image) else im for im in ims]),
        axis = 0,
        trim = trimming,
        power = preference + 1.0,
        weights = np.flip(make_eased_weight_array(len(ims), easing)),
    ), 0.0, 1.0))

def ensure_image_dims(im, mode, size):
    if is_np := isinstance(im, np.ndarray):
        im = Image.fromarray(skimage.util.img_as_ubyte(im))

    if im.mode != mode:
        im = im.convert(mode)

    if im.size != size:
        im = im.resize(size, Image.Resampling.LANCZOS)

    return skimage.util.img_as_float(im) if is_np else im

def generate_noise_image(size, seed = None):
    return np_to_pil(generate_noise((size[1], size[0], 3), seed))

def generate_value_noise_image(size, channels, scale, octaves, lacunarity, persistence, seed = None):
    return np_to_pil(generate_value_noise((size[1], size[0], channels), scale, octaves, lacunarity, persistence, seed))

def load_image(path):
    im = Image.open(path)
    im.load()
    return im

def match_image(im, reference, mode = True, size = True):
    if isinstance(reference, np.ndarray):
        ref_mode = "RGBA" if reference.shape[2] == 4 else "RGB"
        ref_size = (reference.shape[1], reference.shape[0])
    else:
        ref_mode = reference.mode
        ref_size = reference.size

    return ensure_image_dims(im, ref_mode if mode else im.mode, ref_size if size else im.size)

def np_to_pil(npim):
    return Image.fromarray(skimage.util.img_as_ubyte(npim))

def pil_to_np(im):
    return skimage.util.img_as_float(im)

def save_image(im, path, archive_mode = False):
    tmp_path = path.with_suffix(".tmp")

    if path.is_file():
        path.unlink()

    if tmp_path.is_file():
        tmp_path.unlink()

    im.save(tmp_path, "PNG", **(dict(
        optimize = True,
        compress_level = 9,
    ) if archive_mode else {}))
    tmp_path.rename(path)
