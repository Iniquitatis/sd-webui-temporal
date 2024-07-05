from typing import Any

import numpy as np
import scipy
import skimage

from temporal.meta.configurable import StringParam
from temporal.pipeline_modules.filtering import ImageFilter
from temporal.project import Project
from temporal.utils.image import NumpyImage


class CustomCodeFilter(ImageFilter):
    id = "custom_code"
    name = "Custom code"

    code: str = StringParam("Code", value = "output = input", ui_type = "code", language = "python")

    def process(self, npim: NumpyImage, parallel_index: int, project: Project, frame_index: int, seed: int) -> NumpyImage:
        code_globals: dict[str, Any] = dict(
            np = np,
            scipy = scipy,
            skimage = skimage,
            input = npim,
        )
        exec(self.code, code_globals)
        return code_globals.get("output", npim)
