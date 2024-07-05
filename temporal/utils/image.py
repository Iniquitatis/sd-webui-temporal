from pathlib import Path
from typing import Callable, Optional, TypeVar

import numpy as np
import skimage
from PIL import Image
from numpy.typing import NDArray

from temporal.utils.math import lerp


PILImage = Image.Image
NumpyImage = NDArray[np.float_]


T = TypeVar("T", PILImage, NumpyImage)


def alpha_blend(a: NumpyImage, b: NumpyImage) -> NumpyImage:
    if b.shape[-1] == 3:
        return b

    return lerp(a[..., :3], b[..., :3], b[..., [3]])


def apply_channelwise(npim: NumpyImage, func: Callable[[NumpyImage], NumpyImage]) -> NumpyImage:
    return np.stack([func(npim[..., i]) for i in range(npim.shape[-1])], axis = -1)


def checkerboard(shape: tuple[int, ...], cell_size: int, color_1: NDArray[np.float_], color_2: NDArray[np.float_]) -> NumpyImage:
    y, x = np.indices(shape[:2])
    pattern = (x // cell_size + y // cell_size) % 2 == 0
    return np.where(pattern[..., np.newaxis], color_2[..., :shape[-1]], color_1[..., :shape[-1]])


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


def join_hsv_to_rgb(h: NumpyImage, s: NumpyImage, v: NumpyImage) -> NumpyImage:
    return skimage.color.hsv2rgb(np.stack([h, s, v], axis = -1), channel_axis = -1)


def load_image(path: str | Path) -> PILImage:
    im = Image.open(path)
    im.load()
    return im


def match_image(im: T, reference: PILImage | NumpyImage, mode: bool = True, size: bool = True) -> T:
    if isinstance(reference, np.ndarray):
        ref_mode = "RGBA" if reference.shape[-1] == 4 else "RGB"
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
