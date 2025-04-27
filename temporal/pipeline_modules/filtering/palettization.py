from typing import Optional

import numpy as np
from PIL import Image

from temporal.general_data import GeneralData
from temporal.object import Field
from temporal.pipeline_modules.filtering import ImageFilter
from temporal.utils.image import NumpyImage, apply_channelwise, np_to_pil, pil_to_np
from temporal.utils.numpy import FloatType, stretch_array


class PalettizationFilter(ImageFilter):
    name = "Palettization"

    palette: Optional[NumpyImage] = Field(None, name = "Palette", channels = 3)
    stretch: bool = Field(False, name = "Stretch")
    dithering: bool = Field(False, name = "Dithering")

    def process(self, image: NumpyImage, general: GeneralData, iter_index: int, seed: int) -> NumpyImage:
        if self.palette is None:
            return image

        palette_arr = np.array(self.palette, dtype = FloatType).reshape((self.palette.shape[1] * self.palette.shape[0], 3))

        if self.stretch:
            palette_arr = apply_channelwise(palette_arr, lambda x: stretch_array(x, 256))

        palette = Image.new("P", (1, 1))
        palette.putpalette(list(palette_arr.ravel().astype(np.ubyte)), "RGB")

        return pil_to_np(np_to_pil(image).quantize(
            palette = palette,
            colors = palette_arr.size,
            dither = Image.Dither.FLOYDSTEINBERG if self.dithering else Image.Dither.NONE,
        ).convert("RGB"))
