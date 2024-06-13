from abc import abstractmethod
from typing import Any, Optional

import gradio as gr
import numpy as np
import scipy
import skimage
from PIL import Image

from temporal.blend_modes import BLEND_MODES
from temporal.image_mask import ImageMask
from temporal.meta.configurable import ui_param
from temporal.meta.serializable import field
from temporal.pipeline_modules import PipelineModule
from temporal.session import Session
from temporal.utils.image import NumpyImage, alpha_blend, apply_channelwise, get_rgb_array, join_hsv_to_rgb, match_image, np_to_pil, pil_to_np, split_hsv
from temporal.utils.math import lerp, normalize, remap_range
from temporal.utils.numpy import generate_value_noise, saturate_array


class ImageFilter(PipelineModule, abstract = True):
    icon = "\U00002728"

    amount: float = field(1.0)
    amount_relative: bool = field(False)
    blend_mode: str = field("normal")
    mask: ImageMask = field(factory = ImageMask)

    def forward(self, images: list[NumpyImage], session: Session, frame_index: int, seed: int) -> Optional[list[NumpyImage]]:
        return [saturate_array(self._blend(x, self.process(x, seed + i), session)) for i, x in enumerate(images)]

    @abstractmethod
    def process(self, npim: NumpyImage, seed: int) -> NumpyImage:
        raise NotImplementedError

    def _blend(self, npim: NumpyImage, processed: NumpyImage, session: Session) -> NumpyImage:
        if npim is processed:
            return npim

        amount = self.amount * (session.processing.denoising_strength if self.amount_relative else 1.0)

        if amount == 0.0:
            return npim

        processed = BLEND_MODES[self.blend_mode].blend(npim, processed)

        if amount == 1.0 and self.mask.image is None:
            return processed

        if self.mask.image is not None:
            factor = match_image(self.mask.image, npim)

            if self.mask.normalized:
                factor = normalize(factor, factor.min(), factor.max())

            if self.mask.inverted:
                factor = 1.0 - factor

            if self.mask.blurring:
                factor = skimage.filters.gaussian(factor, round(self.mask.blurring), channel_axis = 2)

        else:
            factor = 1.0

        factor *= amount

        return lerp(npim, processed, factor)


class BlurringFilter(ImageFilter):
    id = "blurring"
    name = "Blurring"

    radius: float = ui_param("Radius", gr.Slider, minimum = 0.0, maximum = 50.0, step = 0.1, value = 0.0)

    def process(self, npim: NumpyImage, seed: int) -> NumpyImage:
        return skimage.filters.gaussian(npim, round(self.radius), channel_axis = 2)


class ColorBalancingFilter(ImageFilter):
    id = "color_balancing"
    name = "Color balancing"

    brightness: float = ui_param("Brightness", gr.Slider, minimum = 0.0, maximum = 2.0, step = 0.01, value = 1.0)
    contrast: float = ui_param("Contrast", gr.Slider, minimum = 0.0, maximum = 2.0, step = 0.01, value = 1.0)
    saturation: float = ui_param("Saturation", gr.Slider, minimum = 0.0, maximum = 2.0, step = 0.01, value = 1.0)

    def process(self, npim: NumpyImage, seed: int) -> NumpyImage:
        npim = remap_range(npim, npim.min(), npim.max(), 0.0, self.brightness)

        npim = remap_range(npim, npim.min(), npim.max(), 0.5 - self.contrast / 2, 0.5 + self.contrast / 2)

        h, s, v = split_hsv(npim)
        s[:] = remap_range(s, s.min(), s.max(), s.min(), self.saturation)

        return join_hsv_to_rgb(h, s, v)


class ColorCorrectionFilter(ImageFilter):
    id = "color_correction"
    name = "Color correction"

    image: Optional[NumpyImage] = ui_param("Image", gr.Image, type = "numpy", image_mode = "RGB")
    normalize_contrast: bool = ui_param("Normalize contrast", gr.Checkbox, value = False)
    equalize_histogram: bool = ui_param("Equalize histogram", gr.Checkbox, value = False)

    def process(self, npim: NumpyImage, seed: int) -> NumpyImage:
        if self.image is not None:
            npim = skimage.exposure.match_histograms(npim, match_image(self.image, npim, size = False), channel_axis = 2)

        if self.normalize_contrast:
            npim = skimage.exposure.rescale_intensity(npim)

        if self.equalize_histogram:
            npim = skimage.exposure.equalize_hist(npim)

        return npim


class ColorOverlayFilter(ImageFilter):
    id = "color_overlay"
    name = "Color overlay"

    color: str = ui_param("Color", gr.ColorPicker, value = "#ffffff")

    def process(self, npim: NumpyImage, seed: int) -> NumpyImage:
        color = get_rgb_array(self.color)

        if npim.shape[-1] > color.shape[-1]:
            color = np.stack([color, [1.0]])

        return np.full_like(npim, color)


class CustomCodeFilter(ImageFilter):
    id = "custom_code"
    name = "Custom code"

    code: str = ui_param("Code", gr.Code, language = "python")

    def process(self, npim: NumpyImage, seed: int) -> NumpyImage:
        code_globals: dict[str, Any] = dict(
            np = np,
            scipy = scipy,
            skimage = skimage,
            input = npim,
        )
        exec(self.code, code_globals)
        return code_globals.get("output", npim)


class ImageOverlayFilter(ImageFilter):
    id = "image_overlay"
    name = "Image overlay"

    image: Optional[NumpyImage] = ui_param("Image", gr.Image, type = "numpy", image_mode = "RGBA")
    blurring: float = ui_param("Blurring", gr.Slider, minimum = 0.0, maximum = 50.0, step = 0.1, value = 0.0)

    def process(self, npim: NumpyImage, seed: int) -> NumpyImage:
        if self.image is None:
            return npim

        return match_image(alpha_blend(npim, skimage.filters.gaussian(
            match_image(self.image, npim, mode = False),
            round(self.blurring),
            channel_axis = 2,
        )), npim)


class MedianFilter(ImageFilter):
    id = "median"
    name = "Median"

    radius: int = ui_param("Radius", gr.Slider, precision = 0, minimum = 0, maximum = 50, step = 1, value = 0)
    percentile: float = ui_param("Percentile", gr.Slider, minimum = 0.0, maximum = 100.0, step = 0.1, value = 50.0)

    def process(self, npim: NumpyImage, seed: int) -> NumpyImage:
        footprint = skimage.morphology.disk(self.radius)

        if self.percentile == 50.0:
            filter = lambda x: scipy.ndimage.median_filter(x, footprint = footprint, mode = "nearest")
        else:
            filter = lambda x: scipy.ndimage.percentile_filter(x, self.percentile, footprint = footprint, mode = "nearest")

        return apply_channelwise(npim, filter)


class MorphologyFilter(ImageFilter):
    id = "morphology"
    name = "Morphology"

    mode: str = ui_param("Mode", gr.Dropdown, choices = ["erosion", "dilation", "opening", "closing"], value = "erosion")
    radius: int = ui_param("Radius", gr.Slider, precision = 0, minimum = 0, maximum = 50, step = 1, value = 0)

    def process(self, npim: NumpyImage, seed: int) -> NumpyImage:
        func = (
            skimage.morphology.erosion  if self.mode == "erosion"  else
            skimage.morphology.dilation if self.mode == "dilation" else
            skimage.morphology.opening  if self.mode == "opening"  else
            skimage.morphology.closing  if self.mode == "closing"  else
            lambda image, footprint: image
        )
        footprint = skimage.morphology.disk(self.radius)
        return apply_channelwise(npim, lambda x: func(x, footprint))


class NoiseCompressionFilter(ImageFilter):
    id = "noise_compression"
    name = "Noise compression"

    constant: float = ui_param("Constant", gr.Slider, minimum = 0.0, maximum = 1.0, step = 1e-5, value = 0.0)
    adaptive: float = ui_param("Adaptive", gr.Slider, minimum = 0.0, maximum = 1.0, step = 0.01, value = 0.0)

    def process(self, npim: NumpyImage, seed: int) -> NumpyImage:
        weight = 0.0

        if self.constant > 0.0:
            weight += self.constant

        if self.adaptive > 0.0:
            weight += skimage.restoration.estimate_sigma(npim, average_sigmas = True, channel_axis = 2) * self.adaptive

        return skimage.restoration.denoise_tv_chambolle(npim, weight = max(weight, 1e-5), channel_axis = 2)


class NoiseOverlayFilter(ImageFilter):
    id = "noise_overlay"
    name = "Noise overlay"

    scale: int = ui_param("Scale", gr.Slider, precision = 0, minimum = 1, maximum = 1024, step = 1, value = 1)
    octaves: int = ui_param("Octaves", gr.Slider, precison = 0, minimum = 1, maximum = 10, step = 1, value = 1)
    lacunarity: float = ui_param("Lacunarity", gr.Slider, minimum = 0.01, maximum = 4.0, step = 0.01, value = 2.0)
    persistence: float = ui_param("Persistence", gr.Slider, minimum = 0.0, maximum = 1.0, step = 0.01, value = 0.5)
    seed: int = ui_param("Seed", gr.Number, precision = 0, minimum = 0, step = 1, value = 0)
    use_dynamic_seed: bool = ui_param("Use dynamic seed", gr.Checkbox, value = False)

    def process(self, npim: NumpyImage, seed: int) -> NumpyImage:
        return generate_value_noise(
            npim.shape,
            self.scale,
            self.octaves,
            self.lacunarity,
            self.persistence,
            seed if self.use_dynamic_seed else self.seed,
        )


class PalettizationFilter(ImageFilter):
    id = "palettization"
    name = "Palettization"

    palette: Optional[NumpyImage] = ui_param("Palette", gr.Image, type = "numpy", image_mode = "RGB")
    stretch: bool = ui_param("Stretch", gr.Checkbox, value = False)
    dithering: bool = ui_param("Dithering", gr.Checkbox, value = False)

    def process(self, npim: NumpyImage, seed: int) -> NumpyImage:
        def stretch_array(arr, new_length):
            return np.interp(np.arange(new_length), np.linspace(0, new_length - 1, len(arr)), arr)

        if self.palette is None:
            return npim

        palette_arr = np.array(self.palette, dtype = np.float_).reshape((self.palette.shape[1] * self.palette.shape[0], 3))

        if self.stretch:
            palette_arr = apply_channelwise(palette_arr, lambda x: stretch_array(x, 256))

        palette = Image.new("P", (1, 1))
        palette.putpalette(palette_arr.ravel().astype(np.ubyte), "RGB")

        return pil_to_np(np_to_pil(npim).quantize(
            palette = palette,
            colors = palette_arr.size,
            dither = Image.Dither.FLOYDSTEINBERG if self.dithering else Image.Dither.NONE,
        ).convert("RGB"))


class SharpeningFilter(ImageFilter):
    id = "sharpening"
    name = "Sharpening"

    strength: float = ui_param("Strength", gr.Slider, minimum = 0.0, maximum = 1.0, step = 0.01, value = 0.0)
    radius: float = ui_param("Radius", gr.Slider, minimum = 0.0, maximum = 5.0, step = 0.1, value = 0.0)

    def process(self, npim: NumpyImage, seed: int) -> NumpyImage:
        return skimage.filters.unsharp_mask(npim, self.radius, self.strength, channel_axis = 2)


class SymmetryFilter(ImageFilter):
    id = "symmetry"
    name = "Symmetry"

    horizontal: bool = ui_param("Horizontal", gr.Checkbox, value = False)
    vertical: bool = ui_param("Vertical", gr.Checkbox, value = False)

    def process(self, npim: NumpyImage, seed: int) -> NumpyImage:
        height, width = npim.shape[:2]
        npim = npim.copy()

        if self.horizontal:
            npim[:, width // 2:] = np.flip(npim[:, :width // 2], axis = 1)

        if self.vertical:
            npim[height // 2:, :] = np.flip(npim[:height // 2, :], axis = 0)

        return npim


class TransformationFilter(ImageFilter):
    id = "transformation"
    name = "Transformation"

    translation_x: float = ui_param("Translation X", gr.Slider, minimum = -1.0, maximum = 1.0, step = 0.001, value = 0.0)
    translation_y: float = ui_param("Translation Y", gr.Slider, minimum = -1.0, maximum = 1.0, step = 0.001, value = 0.0)
    rotation: float = ui_param("Rotation", gr.Slider, minimum = -90.0, maximum = 90.0, step = 0.1, value = 0.0)
    scaling: float = ui_param("Scaling", gr.Slider, minimum = 0.0, maximum = 2.0, step = 0.001, value = 1.0)

    def process(self, npim: NumpyImage, seed: int) -> NumpyImage:
        height, width = npim.shape[:2]

        o_transform = skimage.transform.AffineTransform(translation = (-width / 2, -height / 2))
        t_transform = skimage.transform.AffineTransform(translation = (-self.translation_x * width, -self.translation_y * height))
        r_transform = skimage.transform.AffineTransform(rotation = np.deg2rad(self.rotation))
        s_transform = skimage.transform.AffineTransform(scale = self.scaling)

        return skimage.transform.warp(npim, skimage.transform.AffineTransform(t_transform.params @ np.linalg.inv(o_transform.params) @ s_transform.params @ r_transform.params @ o_transform.params).inverse, mode = "symmetric")
