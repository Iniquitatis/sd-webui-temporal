from types import SimpleNamespace

import gradio as gr
import numpy as np
import scipy
import skimage
from PIL import Image, ImageColor

from temporal.collection_utils import reorder_dict
from temporal.func_utils import make_func_registerer
from temporal.image_blending import blend_images
from temporal.image_utils import match_image, np_to_pil, pil_to_np
from temporal.math import lerp, normalize, remap_range
from temporal.numpy_utils import generate_value_noise

PREPROCESSORS, preprocessor = make_func_registerer(name = "", params = [])

def preprocess_image(im, ext_params, seed):
    im = im.convert("RGB")
    npim = pil_to_np(im)

    for key, preprocessor in reorder_dict(PREPROCESSORS, ext_params.preprocessing_order or []).items():
        if not getattr(ext_params, f"{key}_enabled"):
            continue

        npim = _apply_mask(
            npim,
            preprocessor.func(npim, seed, SimpleNamespace(**{x.key: getattr(ext_params, f"{key}_{x.key}") for x in preprocessor.params})),
            getattr(ext_params, f"{key}_amount"),
            getattr(ext_params, f"{key}_blend_mode"),
            getattr(ext_params, f"{key}_mask"),
            getattr(ext_params, f"{key}_mask_normalized"),
            getattr(ext_params, f"{key}_mask_inverted"),
            getattr(ext_params, f"{key}_mask_blurring"),
            im,
        )

    return np_to_pil(np.clip(npim, 0.0, 1.0))

class UIParam:
    def __init__(self, type, key, name, **kwargs):
        self.type = type
        self.key = key
        self.name = name
        self.kwargs = kwargs

@preprocessor("blurring", "Blurring", [
    UIParam(gr.Slider, "radius", "Radius", minimum = 0.0, maximum = 50.0, step = 0.1, value = 0.0),
])
def _(npim, seed, params):
    return skimage.filters.gaussian(npim, params.radius, channel_axis = 2)

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

@preprocessor("color_correction", "Color correction", [
    UIParam(gr.Pil, "image", "Image"),
    UIParam(gr.Checkbox, "normalize_contrast", "Normalize contrast", value = False),
    UIParam(gr.Checkbox, "equalize_histogram", "Equalize histogram", value = False),
])
def _(npim, seed, params):
    if params.image is not None:
        npim = skimage.exposure.match_histograms(npim, pil_to_np(match_image(params.image, npim, size = False)), channel_axis = 2)

    if params.normalize_contrast:
        npim = skimage.exposure.rescale_intensity(npim)

    if params.equalize_histogram:
        npim = skimage.exposure.equalize_hist(npim)

    return npim

@preprocessor("color_overlay", "Color overlay", [
    UIParam(gr.ColorPicker, "color", "Color", value = "#ffffff"),
])
def _(npim, seed, params):
    return np.full_like(npim, np.array(ImageColor.getrgb(params.color)) / 255.0)

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

@preprocessor("image_overlay", "Image overlay", [
    UIParam(gr.Pil, "image", "Image"),
    UIParam(gr.Slider, "blurring", "Blurring", minimum = 0.0, maximum = 50.0, step = 0.1, value = 0.0),
])
def _(npim, seed, params):
    if params.image is None:
        return npim

    return skimage.filters.gaussian(pil_to_np(match_image(params.image, npim)), params.blurring, channel_axis = 2)

@preprocessor("median", "Median", [
    UIParam(gr.Slider, "radius", "Radius", minimum = 0, maximum = 50, step = 1, value = 0),
])
def _(npim, seed, params):
    footprint = skimage.morphology.disk(params.radius)
    return np.stack([
        skimage.filters.median(npim[..., 0], footprint),
        skimage.filters.median(npim[..., 1], footprint),
        skimage.filters.median(npim[..., 2], footprint),
    ], axis = 2)

@preprocessor("morphology", "Morphology", [
    UIParam(gr.Dropdown, "mode", "Mode", choices = ["erosion", "dilation", "opening", "closing"], value = "erosion"),
    UIParam(gr.Slider, "radius", "Radius", minimum = 0, maximum = 50, step = 1, value = 0),
])
def _(npim, seed, params):
    func = (
        skimage.morphology.erosion  if params.mode == "erosion"  else
        skimage.morphology.dilation if params.mode == "dilation" else
        skimage.morphology.opening  if params.mode == "opening"  else
        skimage.morphology.closing  if params.mode == "closing"  else
        lambda image, footprint: image
    )
    footprint = skimage.morphology.disk(params.radius)
    return np.stack([
        func(npim[..., 0], footprint),
        func(npim[..., 1], footprint),
        func(npim[..., 2], footprint),
    ], axis = 2)

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

@preprocessor("noise_overlay", "Noise overlay", [
    UIParam(gr.Slider, "scale", "Scale", minimum = 1, maximum = 1024, step = 1, value = 1),
    UIParam(gr.Slider, "octaves", "Octaves", minimum = 1, maximum = 10, step = 1, value = 1),
    UIParam(gr.Slider, "lacunarity", "Lacunarity", minimum = 0.01, maximum = 4.0, step = 0.01, value = 2.0),
    UIParam(gr.Slider, "persistence", "Persistence", minimum = 0.0, maximum = 1.0, step = 0.01, value = 0.5),
    UIParam(gr.Number, "seed", "Seed", precision = 0, minimum = 0, step = 1, value = 0),
    UIParam(gr.Checkbox, "use_dynamic_seed", "Use dynamic seed", value = False),
])
def _(npim, seed, params):
    return generate_value_noise(
        npim.shape,
        params.scale,
        params.octaves,
        params.lacunarity,
        params.persistence,
        seed if params.use_dynamic_seed else params.seed,
    )

@preprocessor("palettization", "Palettization", [
    UIParam(gr.Pil, "palette", "Palette", image_mode = "RGB"),
    UIParam(gr.Checkbox, "stretch", "Stretch", value = False),
    UIParam(gr.Checkbox, "dithering", "Dithering", value = False),
])
def _(npim, seed, params):
    def stretch_array(arr, new_length):
        return np.interp(np.arange(new_length), np.linspace(0, new_length - 1, len(arr)), arr)

    palette_arr = np.array(params.palette, dtype = np.float_).reshape((params.palette.width * params.palette.height, 3))

    if params.stretch:
        palette_arr = np.stack([
            stretch_array(palette_arr[..., 0], 256),
            stretch_array(palette_arr[..., 1], 256),
            stretch_array(palette_arr[..., 2], 256),
        ], axis = 1)

    palette = Image.new("P", (1, 1))
    palette.putpalette(palette_arr.ravel().astype(np.ubyte), "RGB")

    return pil_to_np(np_to_pil(npim).quantize(
        palette = palette,
        colors = palette_arr.size,
        dither = Image.Dither.FLOYDSTEINBERG if params.dithering else Image.Dither.NONE,
    ).convert("RGB"))

@preprocessor("sharpening", "Sharpening", [
    UIParam(gr.Slider, "strength", "Strength", minimum = 0.0, maximum = 1.0, step = 0.01, value = 0.0),
    UIParam(gr.Slider, "radius", "Radius", minimum = 0.0, maximum = 5.0, step = 0.1, value = 0.0),
])
def _(npim, seed, params):
    return skimage.filters.unsharp_mask(npim, params.radius, params.strength, channel_axis = 2)

@preprocessor("symmetry", "Symmetry")
def _(npim, seed, params):
    _, width = npim.shape[:2]
    npim = npim.copy()
    npim[:, width // 2:] = np.flip(npim[:, :width // 2], axis = 1)
    return npim

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

def _apply_mask(npim, processed, amount, blend_mode, mask, normalized, inverted, blurring, reference):
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
            factor = skimage.filters.gaussian(factor, blurring, channel_axis = 2)

    else:
        factor = 1.0

    factor *= amount

    return lerp(npim, processed, factor)
