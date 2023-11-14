import numpy as np
import scipy
import skimage
from PIL import Image, ImageColor

from temporal.image_blending import blend_images
from temporal.image_utils import match_image
from temporal.math import remap_range

def preprocess_image(im, uv, seed):
    im = im.convert("RGB")
    npim = skimage.img_as_float(im)
    height, width = npim.shape[:2]

    if uv.noise_compression_enabled:
        weight = 0.0

        if uv.noise_compression_constant > 0.0:
            weight += uv.noise_compression_constant

        if uv.noise_compression_adaptive > 0.0:
            weight += skimage.restoration.estimate_sigma(npim, average_sigmas = True, channel_axis = 2) * uv.noise_compression_adaptive

        npim = skimage.restoration.denoise_tv_chambolle(npim, weight = max(weight, 1e-5), channel_axis = 2)

    if uv.color_correction_enabled:
        if uv.color_correction_image is not None:
            npim = skimage.exposure.match_histograms(npim, skimage.img_as_float(match_image(uv.color_correction_image, im, size = False)), channel_axis = 2)

        if uv.normalize_contrast:
            npim = skimage.exposure.rescale_intensity(npim)

    # NOTE: Just matching the mean
    #if uv.luminance_matching_enabled:
    #    luminance = np.mean(skimage.color.rgb2gray(npim, channel_axis = 2))
    #    npim = npim + (uv.luminance_target - luminance)

    if uv.luminance_matching_enabled:
        grayscale = skimage.color.rgb2gray(npim, channel_axis = 2)
        mean, std = np.mean(grayscale), np.std(grayscale)
        npim = uv.luminance_target + (npim - mean) * (uv.level_std_red / std)

    # NOTE: By mean and std
    #if uv.level_matching_enabled:
    #    def adjust_channel(values, target_mean, target_std):
    #        mean, std = np.mean(values), np.std(values)
    #        values[:] = target_mean + (values - mean) * (target_std / std)
    #
    #    adjust_channel(npim[..., 0], uv.level_mean_red, uv.level_std_red)
    #    adjust_channel(npim[..., 1], uv.level_mean_green, uv.level_std_green)
    #    adjust_channel(npim[..., 2], uv.level_mean_blue, uv.level_std_blue)

    # NOTE: To absolute values
    if uv.level_matching_enabled:
        def adjust_channel(values, target):
            values[:] = remap_range(values, values.min(), values.max(), 0.0, target)

        adjust_channel(npim[..., 0], uv.level_mean_red)
        adjust_channel(npim[..., 1], uv.level_mean_green)
        adjust_channel(npim[..., 2], uv.level_mean_blue)

    if uv.level_adjustment_enabled:
        def adjust_channel(values, shadows, midtones, highlights):
            values[:] = remap_range(values, np.min(values), np.max(values), shadows, highlights)
            values += midtones - np.mean(values)

        adjust_channel(npim[..., 0], uv.level_shadows_red, uv.level_midtones_red, uv.level_highlights_red)
        adjust_channel(npim[..., 1], uv.level_shadows_green, uv.level_midtones_green, uv.level_highlights_green)
        adjust_channel(npim[..., 2], uv.level_shadows_blue, uv.level_midtones_blue, uv.level_highlights_blue)

    if uv.color_balancing_enabled:
        npim = remap_range(npim, npim.min(), npim.max(), 0.0, uv.brightness)

        npim = remap_range(npim, npim.min(), npim.max(), 0.5 - uv.contrast / 2, 0.5 + uv.contrast / 2)

        hsv = skimage.color.rgb2hsv(npim, channel_axis = 2)
        s = hsv[..., 1]
        s[:] = remap_range(s, s.min(), s.max(), s.min(), uv.saturation)
        npim = skimage.color.hsv2rgb(hsv)

    if uv.noise_enabled:
        npim = blend_images(npim, np.random.default_rng(seed).uniform(high = 1.0 + np.finfo(npim.dtype).eps, size = npim.shape), uv.noise_mode, uv.noise_amount)

    if uv.modulation_enabled and uv.modulation_image is not None:
        npim = blend_images(npim, skimage.filters.gaussian(skimage.img_as_float(match_image(uv.modulation_image, im)), uv.modulation_blurring, channel_axis = 2), uv.modulation_mode, uv.modulation_amount)

    if uv.tinting_enabled:
        npim = blend_images(npim, np.full_like(npim, np.array(ImageColor.getrgb(uv.tinting_color)) / 255.0), uv.tinting_mode, uv.tinting_amount)

    if uv.sharpening_enabled:
        npim = skimage.filters.unsharp_mask(npim, uv.sharpening_radius, uv.sharpening_amount, channel_axis = 2)

    if uv.transformation_enabled:
        o_transform = skimage.transform.AffineTransform(translation = (-width / 2, -height / 2))
        t_transform = skimage.transform.AffineTransform(translation = (-uv.translation_x * width, -uv.translation_y * height))
        r_transform = skimage.transform.AffineTransform(rotation = np.deg2rad(uv.rotation))
        s_transform = skimage.transform.AffineTransform(scale = uv.scaling)
        npim = skimage.transform.warp(npim, skimage.transform.AffineTransform(t_transform.params @ np.linalg.inv(o_transform.params) @ s_transform.params @ r_transform.params @ o_transform.params).inverse, mode = "symmetric")

    if uv.symmetrize:
        npim[:, width // 2:] = np.flip(npim[:, :width // 2], axis = 1)

    if uv.blurring_enabled:
        npim = skimage.filters.gaussian(npim, uv.blurring_radius, channel_axis = 2)

    if uv.custom_code_enabled:
        code_globals = dict(
            np = np,
            scipy = scipy,
            skimage = skimage,
            input = npim,
        )
        exec(uv.custom_code, code_globals)
        npim = code_globals.get("output", npim)

    return Image.fromarray(skimage.img_as_ubyte(np.clip(npim, 0.0, 1.0)))
