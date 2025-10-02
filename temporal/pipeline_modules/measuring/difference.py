from typing import Optional

import numpy as np

from modules.object import Field
from modules.pipeline_modules.measuring import MeasuringModule
from modules.utils.image import NumpyImage


class DifferenceMeasuringModule(MeasuringModule):
    name = "Difference"
    file_name = "difference"
    channels = [
        ("Minimum", "darkred"),
        ("Mean", "darkgreen"),
        ("Maximum", "darkblue"),
    ]

    last_image: Optional[NumpyImage] = Field(None, flags = {"runtime"})

    def measure(self, image: NumpyImage) -> list[float]:
        if self.last_image is None:
            self.last_image = image.copy()

        diff = np.abs(image - self.last_image)
        self.last_image = image.copy()

        return [diff.min(), diff.mean(), diff.max()]
