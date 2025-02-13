from base64 import b64decode, b64encode
from io import BytesIO
from pathlib import Path
from typing import Callable, Literal, Optional

import numpy as np
import skimage
from PIL import Image
from numpy.typing import NDArray

from temporal.utils.math import lerp
from temporal.utils.numpy import saturate_array


PILImage = Image.Image
NumpyImage = NDArray[np.float64]


def alpha_blend(a: NumpyImage, b: NumpyImage) -> NumpyImage:
    if b.shape[-1] == 3:
        return b

    return lerp(a[..., :3], b[..., :3], b[..., [3]])


def apply_channelwise(npim: NumpyImage, func: Callable[[NumpyImage], NumpyImage]) -> NumpyImage:
    return np.stack([func(npim[..., i]) for i in range(npim.shape[-1])], axis = -1)


def apply_color_matrix(npim: NumpyImage, matrix: NDArray[np.float64], clip: bool = True) -> NumpyImage:
    result = npim.copy()
    result[..., :3] @= matrix.T

    if clip:
        result = saturate_array(result)

    return result


def base64_to_image(data: str) -> NumpyImage:
    png_prefix = "data:image/png;base64,"

    if data.startswith(png_prefix):
        data = data[len(png_prefix):]

    return pil_to_np(Image.open(BytesIO(b64decode(data))))


def ensure_image_dims(npim: NumpyImage, size: Optional[tuple[int, int]] = None, channels: Optional[int] = None) -> NumpyImage:
    npim_height, npim_width, npim_channels = npim.shape

    target_width = size[0] if size is not None else npim_width
    target_height = size[1] if size is not None else npim_height
    target_channels = channels if channels is not None else npim_channels

    if npim_width == target_width and npim_height == target_height and npim_channels == target_channels:
        return npim

    im = np_to_pil(npim)

    if npim_channels != target_channels:
        im = im.convert("RGBA" if target_channels == 4 else "RGB")

    if npim_width != target_width or npim_height != target_height:
        im = im.resize((target_width, target_height), Image.Resampling.LANCZOS)

    return pil_to_np(im)


def image_to_base64(image: NumpyImage, mode: Literal["default", "fast", "archive"] = "default") -> str:
    kwargs = {
        "default": dict(),
        "fast": dict(optimize = True, compress_level = 0),
        "archive": dict(optimize = True, compress_level = 9),
    }

    with BytesIO() as stream:
        np_to_pil(image).save(stream, "PNG", **kwargs[mode])
        return b64encode(stream.getvalue()).decode()


def join_hsv_to_rgb(h: NumpyImage, s: NumpyImage, v: NumpyImage) -> NumpyImage:
    return skimage.color.hsv2rgb(np.stack([h, s, v], axis = -1), channel_axis = -1)


def load_image(path: str | Path) -> PILImage:
    im = Image.open(path)
    im.load()
    return im


def match_image(npim: NumpyImage, reference: NumpyImage, size: bool = True, channels: bool = True) -> NumpyImage:
    return ensure_image_dims(
        npim,
        (reference.shape[1], reference.shape[0]) if size else None,
        reference.shape[2] if channels else None,
    )


def np_to_pil(npim: NumpyImage) -> PILImage:
    return Image.fromarray(skimage.util.img_as_ubyte(npim))


def pil_to_np(im: PILImage) -> NumpyImage:
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
