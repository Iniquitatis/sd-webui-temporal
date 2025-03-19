from typing import Any

import numpy as np
import scipy
import skimage

from temporal.general_data import GeneralData
from temporal.object import Param
from temporal.pipeline_modules.filtering import ImageFilter
from temporal.utils.image import NumpyImage
from temporal.utils.numpy import saturate_array


class CustomCodeFilter(ImageFilter):
    name = "Custom code"

    code: str = Param("Code", value = "output = input", ui_type = "code", language = "python")

    def process(self, image: NumpyImage, general: GeneralData, iter_index: int, seed: int) -> NumpyImage:
        code_globals: dict[str, Any] = dict(
            np = np,
            scipy = scipy,
            skimage = skimage,
            input = image,
        )
        exec(self.code, code_globals)
        return saturate_array(code_globals.get("output", image))
