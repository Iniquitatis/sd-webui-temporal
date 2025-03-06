from io import BytesIO
from pathlib import Path
from typing import Annotated, Callable, Literal, Optional

import numpy as np
import skimage
from PIL import Image

from temporal.utils.base64 import decode, decode_with_mime_type, encode, encode_with_mime_type
from temporal.utils.math import lerp
from temporal.utils.numpy import FloatArray, saturate_array


PILImage = Image.Image
NumpyImage = Annotated[FloatArray, "image"]


def alpha_blend(a: NumpyImage, b: NumpyImage) -> NumpyImage:
    if b.shape[-1] == 3:
        return b

    return lerp(a[..., :3], b[..., :3], b[..., [3]])


def apply_channelwise(image: NumpyImage, func: Callable[[NumpyImage], NumpyImage]) -> NumpyImage:
    return np.stack([func(image[..., i]) for i in range(image.shape[-1])], axis = -1)


def apply_color_matrix(image: NumpyImage, matrix: FloatArray, clip: bool = True) -> NumpyImage:
    result = image.copy()
    result[..., :3] @= matrix.T

    if clip:
        result = saturate_array(result)

    return result


def base64_to_image(text: str, with_mime_type: bool = True) -> NumpyImage:
    if with_mime_type:
        type, subtype, data = decode_with_mime_type(text)
    else:
        type, subtype, data = "image", "png", decode(text)

    if type != "image":
        raise ValueError

    return pil_to_np(Image.open(BytesIO(data), formats = [subtype]))


def ensure_image_dims(image: NumpyImage, size: Optional[tuple[int, int]] = None, channels: Optional[int] = None) -> NumpyImage:
    image_height, image_width, image_channels = image.shape

    target_width = size[0] if size is not None else image_width
    target_height = size[1] if size is not None else image_height
    target_channels = channels if channels is not None else image_channels

    if image_width == target_width and image_height == target_height and image_channels == target_channels:
        return image

    pil_image = np_to_pil(image)

    if image_channels != target_channels:
        pil_image = pil_image.convert("RGBA" if target_channels == 4 else "RGB")

    if image_width != target_width or image_height != target_height:
        pil_image = pil_image.resize((target_width, target_height), Image.Resampling.LANCZOS)

    return pil_to_np(pil_image)


def image_to_base64(image: NumpyImage, with_mime_type: bool = True, mode: Literal["default", "fast", "archive"] = "default") -> str:
    kwargs = {
        "default": dict(),
        "fast": dict(optimize = False, compress_level = 0),
        "archive": dict(optimize = True, compress_level = 9),
    }

    with BytesIO() as stream:
        np_to_pil(image).save(stream, "PNG", **kwargs[mode])

        if with_mime_type:
            return encode_with_mime_type("image", "png", stream.getvalue())
        else:
            return encode(stream.getvalue())


def join_hsv_to_rgb(h: NumpyImage, s: NumpyImage, v: NumpyImage) -> NumpyImage:
    return skimage.color.hsv2rgb(np.stack([h, s, v], axis = -1), channel_axis = -1)


def load_image(path: str | Path) -> PILImage:
    image = Image.open(path)
    image.load()
    return image


def match_image(image: NumpyImage, reference: NumpyImage, size: bool = True, channels: bool = True) -> NumpyImage:
    return ensure_image_dims(
        image,
        (reference.shape[1], reference.shape[0]) if size else None,
        reference.shape[2] if channels else None,
    )


def np_to_pil(image: NumpyImage) -> PILImage:
    return Image.fromarray(skimage.util.img_as_ubyte(image))


def pil_to_np(image: PILImage) -> NumpyImage:
    return skimage.util.img_as_float(image)


def save_image(image: PILImage, path: Path, archive_mode: bool = False) -> None:
    tmp_path = path.with_suffix(".tmp")

    if path.is_file():
        path.unlink()

    if tmp_path.is_file():
        tmp_path.unlink()

    image.save(tmp_path, "PNG", **(dict(
        optimize = True,
        compress_level = 9,
    ) if archive_mode else {}))
    tmp_path.rename(path)


def split_hsv(image: NumpyImage) -> tuple[NumpyImage, NumpyImage, NumpyImage]:
    hsv = skimage.color.rgb2hsv(image, channel_axis = -1)
    return hsv[..., 0], hsv[..., 1], hsv[..., 2]
