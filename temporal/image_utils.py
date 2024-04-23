import numpy as np
import skimage
from PIL import Image, ImageColor

from temporal.numpy_utils import average_array, generate_noise, generate_value_noise, make_eased_weight_array, saturate_array

def apply_channelwise(npim, func):
    return np.stack([func(npim[..., i]) for i in range(npim.shape[-1])], axis = -1)

def average_images(ims, trimming = 0.0, easing = 0.0, preference = 0.0):
    return ims[0] if len(ims) == 1 else np_to_pil(saturate_array(average_array(
        np.stack([pil_to_np(im) for im in ims]),
        axis = 0,
        trim = trimming,
        power = preference + 1.0,
        weights = np.flip(make_eased_weight_array(len(ims), easing)),
    )))

def ensure_image_dims(im, mode = None, size = None):
    if is_np := isinstance(im, np.ndarray):
        im = np_to_pil(im)

    if mode is not None and im.mode != mode:
        im = im.convert(mode)

    if size is not None and im.size != size:
        im = im.resize(size, Image.Resampling.LANCZOS)

    return pil_to_np(im) if is_np else im

def generate_noise_image(size, seed = None):
    return np_to_pil(generate_noise((size[1], size[0], 3), seed))

def generate_value_noise_image(size, channels, scale, octaves, lacunarity, persistence, seed = None):
    return np_to_pil(generate_value_noise((size[1], size[0], channels), scale, octaves, lacunarity, persistence, seed))

def get_rgb_array(color):
    return np.array(ImageColor.getrgb(color), dtype = np.float_) / 255.0

def join_hsv_to_rgb(h, s, v):
    return skimage.color.hsv2rgb(np.stack([h, s, v], axis = -1), channel_axis = -1)

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

    return ensure_image_dims(im, ref_mode if mode else None, ref_size if size else None)

def np_to_pil(npim):
    if isinstance(npim, Image.Image):
        return npim

    return Image.fromarray(skimage.util.img_as_ubyte(npim))

def pil_to_np(im):
    if isinstance(im, np.ndarray):
        return im

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

def split_hsv(npim):
    hsv = skimage.color.rgb2hsv(npim, channel_axis = -1)
    return hsv[..., 0], hsv[..., 1], hsv[..., 2]
