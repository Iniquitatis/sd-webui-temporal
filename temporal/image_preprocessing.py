from types import SimpleNamespace

import gradio as gr
import numpy as np
import scipy
import skimage
from PIL import ImageColor

from temporal.image_blending import BLEND_MODES, blend_images
from temporal.image_utils import match_image, np_to_pil, pil_to_np
from temporal.math import lerp, remap_range

PREPROCESSORS = dict()

def preprocess_image(im, ext_params, seed):
    im = im.convert("RGB")
    npim = pil_to_np(im)

    for key, preprocessor in PREPROCESSORS.items():
        if not getattr(ext_params, f"{key}_enabled"):
            continue

        npim = _apply_mask(
            npim,
            preprocessor.func(npim, seed, SimpleNamespace(**{x.key: getattr(ext_params, f"{key}_{x.key}") for x in preprocessor.params})),
            getattr(ext_params, f"{key}_amount"),
            getattr(ext_params, f"{key}_mask"),
            getattr(ext_params, f"{key}_mask_inverted"),
            getattr(ext_params, f"{key}_mask_blurring"),
            im,
        )

    return np_to_pil(np.clip(npim, 0.0, 1.0))

def iterate_all_preprocessor_keys():
    for key, preprocessor in PREPROCESSORS.items():
        yield f"{key}_enabled"
        yield f"{key}_amount"
        yield f"{key}_amount_relative"

        for param in preprocessor.params:
            yield f"{key}_{param.key}"

        yield f"{key}_mask"
        yield f"{key}_mask_inverted"
        yield f"{key}_mask_blurring"

class UIParam:
    def __init__(self, type, key, name, **kwargs):
        self.type = type
        self.key = key
        self.name = name
        self.kwargs = kwargs

def preprocessor(key, name, params = []):
    def decorator(func):
        PREPROCESSORS[key] = SimpleNamespace(name = name, func = func, params = params)
        return func
    return decorator

@preprocessor("noise_compression", "Noise compression", [
    UIParam(gr.Slider, "constant", "Constant", minimum = 0.0, maximum = 1.0, step = 1e-5, value = 0.0),
    UIParam(gr.Slider, "adaptive", "Adaptive", minimum = 0.0, maximum = 1.0, step = 0.01, value = 0.0),
])
def _(npim, seed, params):
    weight = 0.0

    if params.constant > 0.0:
        weight += params.constant

    if params.adaptive > 0.0:
        weight += skimage.restoration.estimate_sigma(npim, average_sigmas = True, channel_axis = 2) * params.adaptive

    return skimage.restoration.denoise_tv_chambolle(npim, weight = max(weight, 1e-5), channel_axis = 2)

@preprocessor("color_correction", "Color correction", [
    UIParam(gr.Pil, "image", "Image"),
    UIParam(gr.Checkbox, "normalize_contrast", "Normalize contrast", value = False),
])
def _(npim, seed, params):
    if params.image is not None:
        npim = skimage.exposure.match_histograms(npim, pil_to_np(match_image(params.image, npim, size = False)), channel_axis = 2)

    if params.normalize_contrast:
        npim = skimage.exposure.rescale_intensity(npim)

    return npim

@preprocessor("color_balancing", "Color balancing", [
    UIParam(gr.Slider, "brightness", "Brightness", minimum = 0.0, maximum = 2.0, step = 0.01, value = 1.0),
    UIParam(gr.Slider, "contrast", "Contrast", minimum = 0.0, maximum = 2.0, step = 0.01, value = 1.0),
    UIParam(gr.Slider, "saturation", "Saturation", minimum = 0.0, maximum = 2.0, step = 0.01, value = 1.0),
])
def _(npim, seed, params):
    npim = remap_range(npim, npim.min(), npim.max(), 0.0, params.brightness)

    npim = remap_range(npim, npim.min(), npim.max(), 0.5 - params.contrast / 2, 0.5 + params.contrast / 2)

    hsv = skimage.color.rgb2hsv(npim, channel_axis = 2)
    s = hsv[..., 1]
    s[:] = remap_range(s, s.min(), s.max(), s.min(), params.saturation)

    return skimage.color.hsv2rgb(hsv)

@preprocessor("noise", "Noise", [
    UIParam(gr.Dropdown, "mode", "Mode", choices = {k: v["name"] for k, v in BLEND_MODES.items()}, value = next(iter(BLEND_MODES))),
])
def _(npim, seed, params):
    return blend_images(npim, np.random.default_rng(seed).uniform(high = 1.0 + np.finfo(npim.dtype).eps, size = npim.shape), params.mode)

@preprocessor("modulation", "Modulation", [
    UIParam(gr.Dropdown, "mode", "Mode", choices = {k: v["name"] for k, v in BLEND_MODES.items()}, value = next(iter(BLEND_MODES))),
    UIParam(gr.Pil, "image", "Image"),
    UIParam(gr.Slider, "blurring", "Blurring", minimum = 0.0, maximum = 50.0, step = 0.1, value = 0.0),
])
def _(npim, seed, params):
    if params.image is None:
        return npim

    return blend_images(npim, skimage.filters.gaussian(pil_to_np(match_image(params.image, npim)), params.blurring, channel_axis = 2), params.mode)

@preprocessor("tinting", "Tinting", [
    UIParam(gr.Dropdown, "mode", "Mode", choices = {k: v["name"] for k, v in BLEND_MODES.items()}, value = next(iter(BLEND_MODES))),
    UIParam(gr.ColorPicker, "color", "Color", value = "#ffffff"),
])
def _(npim, seed, params):
    return blend_images(npim, np.full_like(npim, np.array(ImageColor.getrgb(params.color)) / 255.0), params.mode)

@preprocessor("sharpening", "Sharpening", [
    UIParam(gr.Slider, "strength", "Strength", minimum = 0.0, maximum = 1.0, step = 0.01, value = 0.0),
    UIParam(gr.Slider, "radius", "Radius", minimum = 0.0, maximum = 5.0, step = 0.1, value = 0.0),
])
def _(npim, seed, params):
    return skimage.filters.unsharp_mask(npim, params.radius, params.strength, channel_axis = 2)

@preprocessor("transformation", "Transformation", [
    UIParam(gr.Slider, "translation_x", "Translation X", minimum = -1.0, maximum = 1.0, step = 0.001, value = 0.0),
    UIParam(gr.Slider, "translation_y", "Translation Y", minimum = -1.0, maximum = 1.0, step = 0.001, value = 0.0),
    UIParam(gr.Slider, "rotation", "Rotation", minimum = -90.0, maximum = 90.0, step = 0.1, value = 0.0),
    UIParam(gr.Slider, "scaling", "Scaling", minimum = 0.0, maximum = 2.0, step = 0.001, value = 1.0),
])
def _(npim, seed, params):
    height, width = npim.shape[:2]

    o_transform = skimage.transform.AffineTransform(translation = (-width / 2, -height / 2))
    t_transform = skimage.transform.AffineTransform(translation = (-params.translation_x * width, -params.translation_y * height))
    r_transform = skimage.transform.AffineTransform(rotation = np.deg2rad(params.rotation))
    s_transform = skimage.transform.AffineTransform(scale = params.scaling)

    return skimage.transform.warp(npim, skimage.transform.AffineTransform(t_transform.params @ np.linalg.inv(o_transform.params) @ s_transform.params @ r_transform.params @ o_transform.params).inverse, mode = "symmetric")

@preprocessor("symmetry", "Symmetry")
def _(npim, seed, params):
    _, width = npim.shape[:2]
    npim = npim.copy()
    npim[:, width // 2:] = np.flip(npim[:, :width // 2], axis = 1)
    return npim

@preprocessor("blurring", "Blurring", [
    UIParam(gr.Slider, "radius", "Radius", minimum = 0.0, maximum = 50.0, step = 0.1, value = 0.0),
])
def _(npim, seed, params):
    return skimage.filters.gaussian(npim, params.radius, channel_axis = 2)

@preprocessor("custom_code", "Custom code", [
    UIParam(gr.Code, "code", "Code", language = "python"),
])
def _(npim, seed, params):
    code_globals = dict(
        np = np,
        scipy = scipy,
        skimage = skimage,
        input = npim,
    )
    exec(params.code, code_globals)
    return code_globals.get("output", npim)

def _apply_mask(npim, processed, amount, mask, inverted, blurring, reference):
    if npim is processed or amount == 0.0:
        return npim

    if amount == 1.0 and mask is None:
        return processed

    if mask:
        factor = pil_to_np(match_image(mask, reference))

        if inverted:
            factor = 1.0 - factor

        if blurring:
            factor = skimage.filters.gaussian(factor, blurring, channel_axis = 2)

    else:
        factor = 1.0

    factor *= amount

    return lerp(npim, processed, factor)
