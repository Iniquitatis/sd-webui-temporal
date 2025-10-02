from typing import Any

import numpy as np
import scipy
import skimage

from temporal.general_data import GeneralData
from temporal.object import Field
from temporal.pipeline_modules.filtering import ImageFilter
from temporal.utils.image import NumpyImage
from temporal.utils.numpy import saturate_array


class CustomCodeFilter(ImageFilter):
    name = "Custom code"

    code: str = Field("output = input", name = "Code", display = "code", language = "python")

    def process(self, image: NumpyImage, general: GeneralData) -> NumpyImage:
        code_globals: dict[str, Any] = dict(
            np = np,
            scipy = scipy,
            skimage = skimage,
            input = image,
        )
        exec(self.code, code_globals)
        return saturate_array(code_globals.get("output", image))
