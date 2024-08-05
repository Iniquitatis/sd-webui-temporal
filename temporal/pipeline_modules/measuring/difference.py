import numpy as np

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

    last_images: list[NumpyImage] = Field(factory = list)

    def reset(self) -> None:
        super().reset()
        self.last_images.clear()

    def measure(self, npim: NumpyImage, parallel_index: int) -> list[float]:
        while parallel_index >= len(self.last_images):
            self.last_images.append(npim.copy())

        diff = np.abs(npim - self.last_images[parallel_index])
        self.last_images[parallel_index] = npim.copy()

        return [diff.min(), diff.mean(), diff.max()]
