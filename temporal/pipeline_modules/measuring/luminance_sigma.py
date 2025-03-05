import numpy as np
import skimage

from temporal.pipeline_modules.measuring import MeasuringModule
from temporal.utils.image import NumpyImage


class LuminanceSigmaMeasuringModule(MeasuringModule):
    name = "Luminance sigma"
    file_name = "luminance_sigma"
    channels = [
        ("Luminance", "gray"),
    ]

    def measure(self, image: NumpyImage) -> list[float]:
        grayscale = skimage.color.rgb2gray(image[..., :3], channel_axis = -1)
        return [float(np.std(grayscale))]
