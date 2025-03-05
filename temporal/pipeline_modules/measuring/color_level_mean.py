import numpy as np

from temporal.pipeline_modules.measuring import MeasuringModule
from temporal.utils.image import NumpyImage


class ColorLevelMeanMeasuringModule(MeasuringModule):
    name = "Color level mean"
    file_name = "color_level_mean"
    channels = [
        ("Red", "darkred"),
        ("Green", "darkgreen"),
        ("Blue", "darkblue"),
    ]

    def measure(self, image: NumpyImage) -> list[float]:
        red, green, blue = image[..., 0], image[..., 1], image[..., 2]
        return [float(np.mean(red)), float(np.mean(green)), float(np.mean(blue))]
