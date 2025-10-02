from typing import Any

import numpy as np
import scipy
import skimage

from modules.general_data import GeneralData
from modules.object import Field
from modules.pipeline_modules.filtering import ImageFilter
from modules.utils.image import NumpyImage
from modules.utils.numpy import saturate_array


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
