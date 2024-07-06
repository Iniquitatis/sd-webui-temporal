import numpy as np

from temporal.pipeline_modules.measuring import MeasuringModule
from temporal.utils.image import NumpyImage


class ColorLevelSigmaMeasuringModule(MeasuringModule):
    name = "Color level sigma"
    file_name = "color_level_sigma"
    channels = [
        ("Red", "darkred"),
        ("Green", "darkgreen"),
        ("Blue", "darkblue"),
    ]

    def measure(self, npim: NumpyImage) -> list[float]:
        red, green, blue = npim[..., 0], npim[..., 1], npim[..., 2]
        return [float(np.std(red)), float(np.std(green)), float(np.std(blue))]
