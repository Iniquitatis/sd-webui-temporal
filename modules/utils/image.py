from io import BytesIO
from pathlib import Path
from typing import Annotated, Callable, Literal, Optional

import numpy as np
import skimage
from PIL import Image
from skimage.transform import AffineTransform as Transform, resize

from modules.utils.base64 import decode, decode_with_mime_type, encode, encode_with_mime_type
from modules.utils.math import lerp
from modules.utils.numpy import FloatArray, saturate_array
from modules.utils.typing import Alias


PILImage = Annotated[Image.Image, Alias("PILImage")]
NumpyImage = Annotated[FloatArray, Alias("NumpyImage")]


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

    if image_width != target_width or image_height != target_height:
        image = resize(image, (target_height, target_width), order = 3, preserve_range = True, anti_aliasing = False)

    if image_channels < target_channels:
        image = np.concatenate((image, np.zeros((target_height, target_width, target_channels - image_channels))), axis = -1)

    if image_channels > target_channels:
        image = image[..., :target_channels]

    return image


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


def make_trs_transform(
    image_size: tuple[int, int],
    *,
    translation: tuple[float, float] = (0.0, 0.0),
    translation_relative: bool = True,
    rotation: float = 0.0,
    scale: float = 1.0,
    origin: tuple[float, float] = (0.5, 0.5),
    origin_relative: bool = True,
) -> Transform:
    abs_translation = tuple(-x for x in translation)

    if translation_relative:
        abs_translation = tuple(x * y for x, y in zip(abs_translation, image_size))

    abs_origin = tuple(-x for x in origin)

    if origin_relative:
        abs_origin = tuple(x * y for x, y in zip(abs_origin, image_size))

    result = Transform()
    result.params @= Transform(translation = abs_translation).params
    result.params @= Transform(translation = abs_origin).inverse.params
    result.params @= Transform(scale = scale).params
    result.params @= Transform(rotation = np.deg2rad(rotation)).params
    result.params @= Transform(translation = abs_origin).params

    return result.inverse


def match_image(image: NumpyImage, reference: NumpyImage, size: bool = True, channels: bool = True) -> NumpyImage:
    return ensure_image_dims(
        image,
        (reference.shape[1], reference.shape[0]) if size else None,
        reference.shape[2] if channels else None,
    )


def np_to_pil(image: NumpyImage) -> PILImage:
    if image.shape[-1] < 3:
        image = ensure_image_dims(image, channels = 3)

    return Image.fromarray(skimage.util.img_as_ubyte(image))


def pil_to_np(image: PILImage) -> NumpyImage:
    match image.mode:
        case "RGB":
            pass
        case "RGBA":
            pass
        case "P":
            image = image.convert("RGB")
        case "PA":
            image = image.convert("RGBA")
        case "L":
            image = image.convert("RGB")
        case "LA":
            image = image.convert("RGBA")
        case _:
            raise Exception(f"Unsupported image mode {image.mode}")

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
