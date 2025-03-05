from typing import Optional

import numpy as np

from temporal.general_data import GeneralData
from temporal.meta.serializable import SerializableField as Field
from temporal.pipeline_modules.measuring import MeasuringModule
from temporal.utils.image import NumpyImage


class DifferenceMeasuringModule(MeasuringModule):
    name = "Difference"
    file_name = "difference"
    channels = [
        ("Minimum", "darkred"),
        ("Mean", "darkgreen"),
        ("Maximum", "darkblue"),
    ]

    last_image: Optional[NumpyImage] = Field(None, variant = "image", flags = {"private"})

    def reset(self, general: GeneralData) -> None:
        super().reset(general)
        self.last_image = None

    def measure(self, npim: NumpyImage) -> list[float]:
        if self.last_image is None:
            self.last_image = npim.copy()

        diff = np.abs(npim - self.last_image)
        self.last_image = npim.copy()

        return [diff.min(), diff.mean(), diff.max()]
