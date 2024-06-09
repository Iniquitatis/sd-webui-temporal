from pathlib import Path
from typing import Callable, Optional, TypeVar

import numpy as np
import skimage
from PIL import Image, ImageColor
from numpy.typing import NDArray

from temporal.utils.numpy import generate_noise, generate_value_noise


PILImage = Image.Image
NumpyImage = NDArray[np.float_]


T = TypeVar("T", PILImage, NumpyImage)


def apply_channelwise(npim: NumpyImage, func: Callable[[NumpyImage], NumpyImage]) -> NumpyImage:
    return np.stack([func(npim[..., i]) for i in range(npim.shape[-1])], axis = -1)


def ensure_image_dims(im: T, mode: Optional[str] = None, size: Optional[tuple[int, int]] = None) -> T:
    if isinstance(im, np.ndarray):
        tmp_im = np_to_pil(im)
    else:
        tmp_im = im

    if mode is not None and tmp_im.mode != mode:
        tmp_im = tmp_im.convert(mode)

    if size is not None and tmp_im.size != size:
        tmp_im = tmp_im.resize(size, Image.Resampling.LANCZOS)

    if isinstance(im, np.ndarray):
        return pil_to_np(tmp_im)
    else:
        return tmp_im


def generate_noise_image(size: tuple[int, int], seed: Optional[int] = None) -> PILImage:
    return np_to_pil(generate_noise((size[1], size[0], 3), seed))


def generate_value_noise_image(size: tuple[int, int], channels: int, scale: float, octaves: int, lacunarity: float, persistence: float, seed: Optional[int] = None) -> PILImage:
    return np_to_pil(generate_value_noise((size[1], size[0], channels), scale, octaves, lacunarity, persistence, seed))


def get_rgb_array(color: str) -> NDArray[np.float_]:
    return np.array(ImageColor.getrgb(color), dtype = np.float_) / 255.0


def join_hsv_to_rgb(h: NumpyImage, s: NumpyImage, v: NumpyImage) -> NumpyImage:
    return skimage.color.hsv2rgb(np.stack([h, s, v], axis = -1), channel_axis = -1)


def load_image(path: str | Path) -> PILImage:
    im = Image.open(path)
    im.load()
    return im


def match_image(im: T, reference: PILImage | NumpyImage, mode: bool = True, size: bool = True) -> T:
    if isinstance(reference, np.ndarray):
        ref_mode = "RGBA" if reference.shape[2] == 4 else "RGB"
        ref_size = (reference.shape[1], reference.shape[0])
    else:
        ref_mode = reference.mode
        ref_size = reference.size

    return ensure_image_dims(im, ref_mode if mode else None, ref_size if size else None)


def np_to_pil(npim: NumpyImage) -> PILImage:
    if isinstance(npim, Image.Image):
        return npim

    return Image.fromarray(skimage.util.img_as_ubyte(npim))


def pil_to_np(im: PILImage) -> NumpyImage:
    if isinstance(im, np.ndarray):
        return im

    return skimage.util.img_as_float(im)


def save_image(im: PILImage, path: Path, archive_mode: bool = False) -> None:
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


def split_hsv(npim: NumpyImage) -> tuple[NumpyImage, NumpyImage, NumpyImage]:
    hsv = skimage.color.rgb2hsv(npim, channel_axis = -1)
    return hsv[..., 0], hsv[..., 1], hsv[..., 2]
