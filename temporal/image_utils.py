import numpy as np
import skimage
from PIL import Image

def ensure_image_dims(im, mode, size):
    if is_np := isinstance(im, np.ndarray):
        im = Image.fromarray(skimage.util.img_as_ubyte(im))

    if im.mode != mode:
        im = im.convert(mode)

    if im.size != size:
        im = im.resize(size, Image.Resampling.LANCZOS)

    return skimage.util.img_as_float(im) if is_np else im

def generate_noise_image(size, seed):
    return Image.fromarray(np.random.default_rng(seed).integers(0, 256, size = (size[1], size[0], 3), dtype = "uint8"))

def load_image(path):
    im = Image.open(path)
    im.load()
    return im

def match_image(im, reference, mode = True, size = True):
    if isinstance(reference, np.ndarray):
        reference = Image.fromarray(skimage.util.img_as_ubyte(reference))

    return ensure_image_dims(im, reference.mode if mode else im.mode, reference.size if size else im.size)

def mean_images(ims):
    return ims[0] if len(ims) == 1 else np_to_pil(np.mean(np.stack([pil_to_np(im) if isinstance(im, Image.Image) else im for im in ims]), axis = 0))

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
