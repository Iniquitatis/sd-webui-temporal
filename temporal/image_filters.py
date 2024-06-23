from abc import abstractmethod
from typing import Any, Optional

import numpy as np
import scipy
import skimage
from PIL import Image

from temporal.blend_modes import BLEND_MODES
from temporal.color import Color
from temporal.image_mask import ImageMask
from temporal.image_source import ImageSource
from temporal.meta.configurable import BoolParam, ColorParam, EnumParam, FloatParam, ImageParam, ImageSourceParam, IntParam, NoiseParam, StringParam
from temporal.meta.serializable import SerializableField as Field
from temporal.noise import Noise
from temporal.pipeline_modules import PipelineModule
from temporal.session import Session
from temporal.utils.image import NumpyImage, alpha_blend, apply_channelwise, join_hsv_to_rgb, match_image, np_to_pil, pil_to_np, split_hsv
from temporal.utils.math import lerp, normalize, remap_range
from temporal.utils.numpy import generate_value_noise, saturate_array


class ImageFilter(PipelineModule, abstract = True):
    icon = "\U00002728"

    amount: float = Field(1.0)
    amount_relative: bool = Field(False)
    blend_mode: str = Field("normal")
    mask: ImageMask = Field(factory = ImageMask)

    def forward(self, images: list[NumpyImage], session: Session, frame_index: int, seed: int) -> Optional[list[NumpyImage]]:
        return [saturate_array(self._blend(x, self.process(x, i, session, frame_index, seed + i), session)) for i, x in enumerate(images)]

    @abstractmethod
    def process(self, npim: NumpyImage, parallel_index: int, session: Session, frame_index: int, seed: int) -> NumpyImage:
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

    radius: float = FloatParam("Radius", minimum = 0.0, maximum = 50.0, step = 0.1, value = 0.0, ui_type = "slider")

    def process(self, npim: NumpyImage, parallel_index: int, session: Session, frame_index: int, seed: int) -> NumpyImage:
        return skimage.filters.gaussian(npim, round(self.radius), channel_axis = 2)


class ColorBalancingFilter(ImageFilter):
    id = "color_balancing"
    name = "Color balancing"

    brightness: float = FloatParam("Brightness", minimum = 0.0, maximum = 2.0, step = 0.01, value = 1.0, ui_type = "slider")
    contrast: float = FloatParam("Contrast", minimum = 0.0, maximum = 2.0, step = 0.01, value = 1.0, ui_type = "slider")
    saturation: float = FloatParam("Saturation", minimum = 0.0, maximum = 2.0, step = 0.01, value = 1.0, ui_type = "slider")

    def process(self, npim: NumpyImage, parallel_index: int, session: Session, frame_index: int, seed: int) -> NumpyImage:
        npim = remap_range(npim, npim.min(), npim.max(), 0.0, self.brightness)

        npim = remap_range(npim, npim.min(), npim.max(), 0.5 - self.contrast / 2, 0.5 + self.contrast / 2)

        h, s, v = split_hsv(npim)
        s[:] = remap_range(s, s.min(), s.max(), s.min(), self.saturation)

        return join_hsv_to_rgb(h, s, v)


class ColorCorrectionFilter(ImageFilter):
    id = "color_correction"
    name = "Color correction"

    source: ImageSource = ImageSourceParam("Image source", channels = 3)
    normalize_contrast: bool = BoolParam("Normalize contrast", value = False)
    equalize_histogram: bool = BoolParam("Equalize histogram", value = False)

    def process(self, npim: NumpyImage, parallel_index: int, session: Session, frame_index: int, seed: int) -> NumpyImage:
        if (image := self.source.get_image(session.processing.init_images[parallel_index], frame_index - 1)) is not None:
            npim = skimage.exposure.match_histograms(npim, match_image(image, npim, size = False), channel_axis = 2)

        if self.normalize_contrast:
            npim = skimage.exposure.rescale_intensity(npim)

        if self.equalize_histogram:
            npim = skimage.exposure.equalize_hist(npim)

        return npim


class ColorOverlayFilter(ImageFilter):
    id = "color_overlay"
    name = "Color overlay"

    color: Color = ColorParam("Color", channels = 3)

    def process(self, npim: NumpyImage, parallel_index: int, session: Session, frame_index: int, seed: int) -> NumpyImage:
        return np.full_like(npim, self.color.to_numpy(npim.shape[-1]))


class CustomCodeFilter(ImageFilter):
    id = "custom_code"
    name = "Custom code"

    code: str = StringParam("Code", value = "output = input", ui_type = "code", language = "python")

    def process(self, npim: NumpyImage, parallel_index: int, session: Session, frame_index: int, seed: int) -> NumpyImage:
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

    source: ImageSource = ImageSourceParam("Image source", channels = 4)
    blurring: float = FloatParam("Blurring", minimum = 0.0, maximum = 50.0, step = 0.1, value = 0.0, ui_type = "slider")

    def process(self, npim: NumpyImage, parallel_index: int, session: Session, frame_index: int, seed: int) -> NumpyImage:
        if (image := self.source.get_image(session.processing.init_images[parallel_index], frame_index - 1)) is None:
            return npim

        return match_image(alpha_blend(npim, skimage.filters.gaussian(
            match_image(image, npim, mode = False),
            round(self.blurring),
            channel_axis = 2,
        )), npim)


class MedianFilter(ImageFilter):
    id = "median"
    name = "Median"

    radius: int = IntParam("Radius", minimum = 0, maximum = 50, step = 1, value = 0, ui_type = "slider")
    percentile: float = FloatParam("Percentile", minimum = 0.0, maximum = 100.0, step = 0.1, value = 50.0, ui_type = "slider")

    def process(self, npim: NumpyImage, parallel_index: int, session: Session, frame_index: int, seed: int) -> NumpyImage:
        footprint = skimage.morphology.disk(self.radius)

        if self.percentile == 50.0:
            filter = lambda x: scipy.ndimage.median_filter(x, footprint = footprint, mode = "nearest")
        else:
            filter = lambda x: scipy.ndimage.percentile_filter(x, self.percentile, footprint = footprint, mode = "nearest")

        return apply_channelwise(npim, filter)


class MorphologyFilter(ImageFilter):
    id = "morphology"
    name = "Morphology"

    mode: str = EnumParam("Mode", choices = ["erosion", "dilation", "opening", "closing"], value = "erosion", ui_type = "menu")
    radius: int = IntParam("Radius", minimum = 0, maximum = 50, step = 1, value = 0, ui_type = "slider")

    def process(self, npim: NumpyImage, parallel_index: int, session: Session, frame_index: int, seed: int) -> NumpyImage:
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

    constant: float = FloatParam("Constant", minimum = 0.0, maximum = 1.0, step = 1e-5, value = 0.0, ui_type = "slider")
    adaptive: float = FloatParam("Adaptive", minimum = 0.0, maximum = 1.0, step = 0.01, value = 0.0, ui_type = "slider")

    def process(self, npim: NumpyImage, parallel_index: int, session: Session, frame_index: int, seed: int) -> NumpyImage:
        weight = 0.0

        if self.constant > 0.0:
            weight += self.constant

        if self.adaptive > 0.0:
            weight += skimage.restoration.estimate_sigma(npim, average_sigmas = True, channel_axis = 2) * self.adaptive

        return skimage.restoration.denoise_tv_chambolle(npim, weight = max(weight, 1e-5), channel_axis = 2)


class NoiseOverlayFilter(ImageFilter):
    id = "noise_overlay"
    name = "Noise overlay"

    noise: Noise = NoiseParam("Noise")
    use_dynamic_seed: bool = BoolParam("Use dynamic seed", value = False)

    def process(self, npim: NumpyImage, parallel_index: int, session: Session, frame_index: int, seed: int) -> NumpyImage:
        return generate_value_noise(
            npim.shape,
            self.noise.scale,
            self.noise.octaves,
            self.noise.lacunarity,
            self.noise.persistence,
            seed if self.use_dynamic_seed else self.noise.seed,
        )


class PalettizationFilter(ImageFilter):
    id = "palettization"
    name = "Palettization"

    palette: Optional[NumpyImage] = ImageParam("Palette", channels = 3)
    stretch: bool = BoolParam("Stretch", value = False)
    dithering: bool = BoolParam("Dithering", value = False)

    def process(self, npim: NumpyImage, parallel_index: int, session: Session, frame_index: int, seed: int) -> NumpyImage:
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

    strength: float = FloatParam("Strength", minimum = 0.0, maximum = 1.0, step = 0.01, value = 0.0, ui_type = "slider")
    radius: float = FloatParam("Radius", minimum = 0.0, maximum = 5.0, step = 0.1, value = 0.0, ui_type = "slider")

    def process(self, npim: NumpyImage, parallel_index: int, session: Session, frame_index: int, seed: int) -> NumpyImage:
        return skimage.filters.unsharp_mask(npim, self.radius, self.strength, channel_axis = 2)


class SymmetryFilter(ImageFilter):
    id = "symmetry"
    name = "Symmetry"

    horizontal: bool = BoolParam("Horizontal", value = False)
    vertical: bool = BoolParam("Vertical", value = False)

    def process(self, npim: NumpyImage, parallel_index: int, session: Session, frame_index: int, seed: int) -> NumpyImage:
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

    translation_x: float = FloatParam("Translation X", minimum = -1.0, maximum = 1.0, step = 0.001, value = 0.0, ui_type = "slider")
    translation_y: float = FloatParam("Translation Y", minimum = -1.0, maximum = 1.0, step = 0.001, value = 0.0, ui_type = "slider")
    rotation: float = FloatParam("Rotation", minimum = -90.0, maximum = 90.0, step = 0.1, value = 0.0, ui_type = "slider")
    scaling: float = FloatParam("Scaling", minimum = 0.0, maximum = 2.0, step = 0.001, value = 1.0, ui_type = "slider")

    def process(self, npim: NumpyImage, parallel_index: int, session: Session, frame_index: int, seed: int) -> NumpyImage:
        height, width = npim.shape[:2]

        o_transform = skimage.transform.AffineTransform(translation = (-width / 2, -height / 2))
        t_transform = skimage.transform.AffineTransform(translation = (-self.translation_x * width, -self.translation_y * height))
        r_transform = skimage.transform.AffineTransform(rotation = np.deg2rad(self.rotation))
        s_transform = skimage.transform.AffineTransform(scale = self.scaling)

        return skimage.transform.warp(npim, skimage.transform.AffineTransform(t_transform.params @ np.linalg.inv(o_transform.params) @ s_transform.params @ r_transform.params @ o_transform.params).inverse, mode = "symmetric")
