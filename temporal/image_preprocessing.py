from types import SimpleNamespace

import gradio as gr
import numpy as np
import scipy
import skimage
from PIL import Image

from temporal.collection_utils import reorder_dict
from temporal.func_utils import make_func_registerer
from temporal.image_blending import blend_images
from temporal.image_utils import apply_channelwise, ensure_image_dims, get_rgb_array, join_hsv_to_rgb, match_image, np_to_pil, pil_to_np, split_hsv
from temporal.math import lerp, normalize, remap_range
from temporal.numpy_utils import generate_value_noise, match_array_dimensions, saturate_array

PREPROCESSORS, preprocessor = make_func_registerer(name = "", params = [])

def preprocess_image(im, ext_params, seed):
    im = ensure_image_dims(im, "RGB")
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

    return np_to_pil(saturate_array(npim))

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

    h, s, v = split_hsv(npim)
    s[:] = remap_range(s, s.min(), s.max(), s.min(), params.saturation)

    return join_hsv_to_rgb(h, s, v)

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
    return np.full_like(npim, get_rgb_array(params.color))

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
    UIParam(gr.Slider, "percentile", "Percentile", minimum = 0.0, maximum = 100.0, step = 0.1, value = 50.0),
])
def _(npim, seed, params):
    footprint = skimage.morphology.disk(params.radius)

    if params.percentile == 50.0:
        filter = lambda x: scipy.ndimage.median_filter(x, footprint = footprint, mode = "nearest")
    else:
        filter = lambda x: scipy.ndimage.percentile_filter(x, params.percentile, footprint = footprint, mode = "nearest")

    return apply_channelwise(npim, filter)

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
    return apply_channelwise(npim, lambda x: func(x, footprint))

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

@preprocessor("outline", "Outline", [
    UIParam(gr.Dropdown, "algorithm", "Algorithm", choices = ["sobel", "scharr", "prewitt", "roberts_cross"], value = "sobel"),
    UIParam(gr.Slider, "thickness", "Thickness", minimum = 1, maximum = 50, step = 1, value = 1),
    UIParam(gr.Checkbox, "rescale", "Rescale", value = False),
    UIParam(gr.ColorPicker, "color", "Color", value = "#000000"),
])
def _(npim, seed, params):
    func = (
        skimage.filters.sobel   if params.algorithm == "sobel"         else
        skimage.filters.scharr  if params.algorithm == "scharr"        else
        skimage.filters.prewitt if params.algorithm == "prewitt"       else
        skimage.filters.roberts if params.algorithm == "roberts_cross" else
        lambda image: image
    )

    outline = saturate_array(func(skimage.color.rgb2gray(npim)))

    if params.thickness > 1:
        outline = skimage.morphology.dilation(outline, skimage.morphology.disk(params.thickness - 1))

    if params.rescale:
        outline = skimage.exposure.rescale_intensity(outline)

    return lerp(npim, get_rgb_array(params.color), match_array_dimensions(outline, npim, 0))

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
        palette_arr = apply_channelwise(palette_arr, lambda x: stretch_array(x, 256))

    palette = Image.new("P", (1, 1))
    palette.putpalette(palette_arr.ravel().astype(np.ubyte), "RGB")

    return pil_to_np(np_to_pil(npim).quantize(
        palette = palette,
        colors = palette_arr.size,
        dither = Image.Dither.FLOYDSTEINBERG if params.dithering else Image.Dither.NONE,
    ).convert("RGB"))

@preprocessor("pixelization", "Pixelization", [
    UIParam(gr.Number, "pixel_size", "Pixel size", minimum = 1, step = 1, value = 1),
])
def _(npim, seed, params):
    def resize(npim, size):
        return skimage.transform.resize(npim, output_shape = size, order = 0, anti_aliasing = False)

    return resize(resize(npim, (npim.shape[0] // params.pixel_size, npim.shape[1] // params.pixel_size)), npim.shape[:2])

@preprocessor("sharpening", "Sharpening", [
    UIParam(gr.Slider, "strength", "Strength", minimum = 0.0, maximum = 1.0, step = 0.01, value = 0.0),
    UIParam(gr.Slider, "radius", "Radius", minimum = 0.0, maximum = 5.0, step = 0.1, value = 0.0),
])
def _(npim, seed, params):
    return skimage.filters.unsharp_mask(npim, params.radius, params.strength, channel_axis = 2)

@preprocessor("symmetry", "Symmetry", [
    UIParam(gr.Checkbox, "horizontal", "Horizontal", value = False),
    UIParam(gr.Checkbox, "vertical", "Vertical", value = False),
])
def _(npim, seed, params):
    height, width = npim.shape[:2]
    npim = npim.copy()

    if params.horizontal:
        npim[:, width // 2:] = np.flip(npim[:, :width // 2], axis = 1)

    if params.vertical:
        npim[height // 2:, :] = np.flip(npim[:height // 2, :], axis = 0)

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
