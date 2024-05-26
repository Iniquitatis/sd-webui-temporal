from abc import abstractmethod
from types import SimpleNamespace
from typing import Any, Optional, Type

import gradio as gr
import numpy as np
import scipy
import skimage
from PIL import Image

from temporal.image_blending import blend_images
from temporal.meta.configurable import Configurable, UIParam
from temporal.utils.collection import reorder_dict
from temporal.utils.image import NumpyImage, PILImage, apply_channelwise, ensure_image_dims, get_rgb_array, join_hsv_to_rgb, match_image, np_to_pil, pil_to_np, split_hsv
from temporal.utils.math import lerp, normalize, remap_range
from temporal.utils.numpy import generate_value_noise, saturate_array


PREPROCESSORS: dict[str, Type["Preprocessor"]] = {}


def preprocess_image(im: PILImage, ext_params: SimpleNamespace, seed: int) -> PILImage:
    im = ensure_image_dims(im, "RGB")
    npim = pil_to_np(im)

    for id, preprocessor in reorder_dict(PREPROCESSORS, ext_params.preprocessing_order or []).items():
        if not getattr(ext_params, f"{id}_enabled"):
            continue

        npim = _apply_mask(
            npim,
            preprocessor.preprocess(npim, seed, SimpleNamespace(**{x.id: getattr(ext_params, f"{id}_{x.id}") for x in preprocessor.params.values()})),
            getattr(ext_params, f"{id}_amount"),
            getattr(ext_params, f"{id}_blend_mode"),
            getattr(ext_params, f"{id}_mask"),
            getattr(ext_params, f"{id}_mask_normalized"),
            getattr(ext_params, f"{id}_mask_inverted"),
            getattr(ext_params, f"{id}_mask_blurring"),
            im,
        )

    return np_to_pil(saturate_array(npim))


class Preprocessor(Configurable, abstract = True):
    store = PREPROCESSORS

    @staticmethod
    @abstractmethod
    def preprocess(npim: NumpyImage, seed: int, params: SimpleNamespace) -> NumpyImage:
        raise NotImplementedError


class BlurringPreprocessor(Preprocessor):
    id = "blurring"
    name = "Blurring"
    params = {
        "radius": UIParam("Radius", gr.Slider, minimum = 0.0, maximum = 50.0, step = 0.1, value = 0.0),
    }

    @staticmethod
    def preprocess(npim: NumpyImage, seed: int, params: SimpleNamespace) -> NumpyImage:
        return skimage.filters.gaussian(npim, params.radius, channel_axis = 2)


class ColorBalancingPreprocessor(Preprocessor):
    id = "color_balancing"
    name = "Color balancing"
    params = {
        "brightness": UIParam("Brightness", gr.Slider, minimum = 0.0, maximum = 2.0, step = 0.01, value = 1.0),
        "contrast": UIParam("Contrast", gr.Slider, minimum = 0.0, maximum = 2.0, step = 0.01, value = 1.0),
        "saturation": UIParam("Saturation", gr.Slider, minimum = 0.0, maximum = 2.0, step = 0.01, value = 1.0),
    }

    @staticmethod
    def preprocess(npim: NumpyImage, seed: int, params: SimpleNamespace) -> NumpyImage:
        npim = remap_range(npim, npim.min(), npim.max(), 0.0, params.brightness)

        npim = remap_range(npim, npim.min(), npim.max(), 0.5 - params.contrast / 2, 0.5 + params.contrast / 2)

        h, s, v = split_hsv(npim)
        s[:] = remap_range(s, s.min(), s.max(), s.min(), params.saturation)

        return join_hsv_to_rgb(h, s, v)


class ColorCorrectionPreprocessor(Preprocessor):
    id = "color_correction"
    name = "Color correction"
    params = {
        "image": UIParam("Image", gr.Pil),
        "normalize_contrast": UIParam("Normalize contrast", gr.Checkbox, value = False),
        "equalize_histogram": UIParam("Equalize histogram", gr.Checkbox, value = False),
    }

    @staticmethod
    def preprocess(npim: NumpyImage, seed: int, params: SimpleNamespace) -> NumpyImage:
        if params.image is not None:
            npim = skimage.exposure.match_histograms(npim, pil_to_np(match_image(params.image, npim, size = False)), channel_axis = 2)

        if params.normalize_contrast:
            npim = skimage.exposure.rescale_intensity(npim)

        if params.equalize_histogram:
            npim = skimage.exposure.equalize_hist(npim)

        return npim


class ColorOverlayPreprocessor(Preprocessor):
    id = "color_overlay"
    name = "Color overlay"
    params = {
        "color": UIParam("Color", gr.ColorPicker, value = "#ffffff"),
    }

    @staticmethod
    def preprocess(npim: NumpyImage, seed: int, params: SimpleNamespace) -> NumpyImage:
        return np.full_like(npim, get_rgb_array(params.color))


class CustomCodePreprocessor(Preprocessor):
    id = "custom_code"
    name = "Custom code"
    params = {
        "code": UIParam("Code", gr.Code, language = "python"),
    }

    @staticmethod
    def preprocess(npim: NumpyImage, seed: int, params: SimpleNamespace) -> NumpyImage:
        code_globals: dict[str, Any] = dict(
            np = np,
            scipy = scipy,
            skimage = skimage,
            input = npim,
        )
        exec(params.code, code_globals)
        return code_globals.get("output", npim)


class ImageOverlayPreprocessor(Preprocessor):
    id = "image_overlay"
    name = "Image overlay"
    params = {
        "image": UIParam("Image", gr.Pil),
        "blurring": UIParam("Blurring", gr.Slider, minimum = 0.0, maximum = 50.0, step = 0.1, value = 0.0),
    }

    @staticmethod
    def preprocess(npim: NumpyImage, seed: int, params: SimpleNamespace) -> NumpyImage:
        if params.image is None:
            return npim

        return skimage.filters.gaussian(pil_to_np(match_image(params.image, npim)), params.blurring, channel_axis = 2)


class MedianPreprocessor(Preprocessor):
    id = "median"
    name = "Median"
    params = {
        "radius": UIParam("Radius", gr.Slider, minimum = 0, maximum = 50, step = 1, value = 0),
        "percentile": UIParam("Percentile", gr.Slider, minimum = 0.0, maximum = 100.0, step = 0.1, value = 50.0),
    }

    @staticmethod
    def preprocess(npim: NumpyImage, seed: int, params: SimpleNamespace) -> NumpyImage:
        footprint = skimage.morphology.disk(params.radius)

        if params.percentile == 50.0:
            filter = lambda x: scipy.ndimage.median_filter(x, footprint = footprint, mode = "nearest")
        else:
            filter = lambda x: scipy.ndimage.percentile_filter(x, params.percentile, footprint = footprint, mode = "nearest")

        return apply_channelwise(npim, filter)


class MorphologyPreprocessor(Preprocessor):
    id = "morphology"
    name = "Morphology"
    params = {
        "mode": UIParam("Mode", gr.Dropdown, choices = ["erosion", "dilation", "opening", "closing"], value = "erosion"),
        "radius": UIParam("Radius", gr.Slider, minimum = 0, maximum = 50, step = 1, value = 0),
    }

    @staticmethod
    def preprocess(npim: NumpyImage, seed: int, params: SimpleNamespace) -> NumpyImage:
        func = (
            skimage.morphology.erosion  if params.mode == "erosion"  else
            skimage.morphology.dilation if params.mode == "dilation" else
            skimage.morphology.opening  if params.mode == "opening"  else
            skimage.morphology.closing  if params.mode == "closing"  else
            lambda image, footprint: image
        )
        footprint = skimage.morphology.disk(params.radius)
        return apply_channelwise(npim, lambda x: func(x, footprint))


class NoiseCompressionPreprocessor(Preprocessor):
    id = "noise_compression"
    name = "Noise compression"
    params = {
        "constant": UIParam("Constant", gr.Slider, minimum = 0.0, maximum = 1.0, step = 1e-5, value = 0.0),
        "adaptive": UIParam("Adaptive", gr.Slider, minimum = 0.0, maximum = 1.0, step = 0.01, value = 0.0),
    }

    @staticmethod
    def preprocess(npim: NumpyImage, seed: int, params: SimpleNamespace) -> NumpyImage:
        weight = 0.0

        if params.constant > 0.0:
            weight += params.constant

        if params.adaptive > 0.0:
            weight += skimage.restoration.estimate_sigma(npim, average_sigmas = True, channel_axis = 2) * params.adaptive

        return skimage.restoration.denoise_tv_chambolle(npim, weight = max(weight, 1e-5), channel_axis = 2)


class NoiseOverlayPreprocessor(Preprocessor):
    id = "noise_overlay"
    name = "Noise overlay"
    params = {
        "scale": UIParam("Scale", gr.Slider, minimum = 1, maximum = 1024, step = 1, value = 1),
        "octaves": UIParam("Octaves", gr.Slider, minimum = 1, maximum = 10, step = 1, value = 1),
        "lacunarity": UIParam("Lacunarity", gr.Slider, minimum = 0.01, maximum = 4.0, step = 0.01, value = 2.0),
        "persistence": UIParam("Persistence", gr.Slider, minimum = 0.0, maximum = 1.0, step = 0.01, value = 0.5),
        "seed": UIParam("Seed", gr.Number, precision = 0, minimum = 0, step = 1, value = 0),
        "use_dynamic_seed": UIParam("Use dynamic seed", gr.Checkbox, value = False),
    }

    @staticmethod
    def preprocess(npim: NumpyImage, seed: int, params: SimpleNamespace) -> NumpyImage:
        return generate_value_noise(
            npim.shape,
            params.scale,
            params.octaves,
            params.lacunarity,
            params.persistence,
            seed if params.use_dynamic_seed else params.seed,
        )


class PalettizationPreprocessor(Preprocessor):
    id = "palettization"
    name = "Palettization"
    params = {
        "palette": UIParam("Palette", gr.Pil, image_mode = "RGB"),
        "stretch": UIParam("Stretch", gr.Checkbox, value = False),
        "dithering": UIParam("Dithering", gr.Checkbox, value = False),
    }

    @staticmethod
    def preprocess(npim: NumpyImage, seed: int, params: SimpleNamespace) -> NumpyImage:
        def stretch_array(arr, new_length):
            return np.interp(np.arange(new_length), np.linspace(0, new_length - 1, len(arr)), arr)

        palette_arr = np.array(params.palette, dtype = np.float_).reshape((params.palette.width * params.palette.height, 3))

        if params.stretch:
            palette_arr = apply_channelwise(palette_arr, lambda x: stretch_array(x, 256))

        palette = Image.new("P", (1, 1))
        palette.putpalette(palette_arr.ravel().astype(np.ubyte), "RGB")

        return pil_to_np(np_to_pil(npim).quantize(
            palette = palette,
            colors = palette_arr.size,
            dither = Image.Dither.FLOYDSTEINBERG if params.dithering else Image.Dither.NONE,
        ).convert("RGB"))


class SharpeningPreprocessor(Preprocessor):
    id = "sharpening"
    name = "Sharpening"
    params = {
        "strength": UIParam("Strength", gr.Slider, minimum = 0.0, maximum = 1.0, step = 0.01, value = 0.0),
        "radius": UIParam("Radius", gr.Slider, minimum = 0.0, maximum = 5.0, step = 0.1, value = 0.0),
    }

    @staticmethod
    def preprocess(npim: NumpyImage, seed: int, params: SimpleNamespace) -> NumpyImage:
        return skimage.filters.unsharp_mask(npim, params.radius, params.strength, channel_axis = 2)


class SymmetryPreprocessor(Preprocessor):
    id = "symmetry"
    name = "Symmetry"
    params = {
        "horizontal": UIParam("Horizontal", gr.Checkbox, value = False),
        "vertical": UIParam("Vertical", gr.Checkbox, value = False),
    }

    @staticmethod
    def preprocess(npim: NumpyImage, seed: int, params: SimpleNamespace) -> NumpyImage:
        height, width = npim.shape[:2]
        npim = npim.copy()

        if params.horizontal:
            npim[:, width // 2:] = np.flip(npim[:, :width // 2], axis = 1)

        if params.vertical:
            npim[height // 2:, :] = np.flip(npim[:height // 2, :], axis = 0)

        return npim


class TransformationPreprocessor(Preprocessor):
    id = "transformation"
    name = "Transformation"
    params = {
        "translation_x": UIParam("Translation X", gr.Slider, minimum = -1.0, maximum = 1.0, step = 0.001, value = 0.0),
        "translation_y": UIParam("Translation Y", gr.Slider, minimum = -1.0, maximum = 1.0, step = 0.001, value = 0.0),
        "rotation": UIParam("Rotation", gr.Slider, minimum = -90.0, maximum = 90.0, step = 0.1, value = 0.0),
        "scaling": UIParam("Scaling", gr.Slider, minimum = 0.0, maximum = 2.0, step = 0.001, value = 1.0),
    }

    @staticmethod
    def preprocess(npim: NumpyImage, seed: int, params: SimpleNamespace) -> NumpyImage:
        height, width = npim.shape[:2]

        o_transform = skimage.transform.AffineTransform(translation = (-width / 2, -height / 2))
        t_transform = skimage.transform.AffineTransform(translation = (-params.translation_x * width, -params.translation_y * height))
        r_transform = skimage.transform.AffineTransform(rotation = np.deg2rad(params.rotation))
        s_transform = skimage.transform.AffineTransform(scale = params.scaling)

        return skimage.transform.warp(npim, skimage.transform.AffineTransform(t_transform.params @ np.linalg.inv(o_transform.params) @ s_transform.params @ r_transform.params @ o_transform.params).inverse, mode = "symmetric")


def _apply_mask(npim: NumpyImage, processed: NumpyImage, amount: float, blend_mode: str, mask: Optional[PILImage], normalized: bool, inverted: bool, blurring: float, reference: PILImage) -> NumpyImage:
    if npim is processed or amount == 0.0:
        return npim

    processed = blend_images(npim, processed, blend_mode)

    if amount == 1.0 and mask is None:
        return processed

    if mask:
        factor = pil_to_np(match_image(mask, reference))

        if normalized:
            factor = normalize(factor, factor.min(), factor.max())

        if inverted:
            factor = 1.0 - factor

        if blurring:
            factor = skimage.filters.gaussian(factor, round(blurring), channel_axis = 2)

    else:
        factor = 1.0

    factor *= amount

    return lerp(npim, processed, factor)
