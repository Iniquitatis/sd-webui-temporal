from typing import Optional

import numpy as np
from PIL import Image
from numpy.typing import NDArray

from temporal.meta.configurable import BoolParam, ImageParam
from temporal.pipeline_modules.filtering import ImageFilter
from temporal.project import Project
from temporal.utils.image import NumpyImage, apply_channelwise, np_to_pil, pil_to_np


class PalettizationFilter(ImageFilter):
    name = "Palettization"

    palette: Optional[NumpyImage] = ImageParam("Palette", channels = 3)
    stretch: bool = BoolParam("Stretch", value = False)
    dithering: bool = BoolParam("Dithering", value = False)

    def process(self, npim: NumpyImage, parallel_index: int, project: Project, frame_index: int, seed: int) -> NumpyImage:
        def stretch_array(arr: NDArray[np.float_], new_length: int) -> NDArray[np.float_]:
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
