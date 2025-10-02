import numpy as np

from modules.pipeline_modules.measuring import MeasuringModule
from modules.utils.image import NumpyImage


class ColorLevelSigmaMeasuringModule(MeasuringModule):
    name = "Color level sigma"
    file_name = "color_level_sigma"
    channels = [
        ("Red", "darkred"),
        ("Green", "darkgreen"),
        ("Blue", "darkblue"),
    ]

    def measure(self, image: NumpyImage) -> list[float]:
        red, green, blue = image[..., 0], image[..., 1], image[..., 2]
        return [float(np.std(red)), float(np.std(green)), float(np.std(blue))]
