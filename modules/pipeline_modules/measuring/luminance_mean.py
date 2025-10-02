import numpy as np
import skimage

from modules.pipeline_modules.measuring import MeasuringModule
from modules.utils.image import NumpyImage


class LuminanceMeanMeasuringModule(MeasuringModule):
    name = "Luminance mean"
    file_name = "luminance_mean"
    channels = [
        ("Luminance", "gray"),
    ]

    def measure(self, image: NumpyImage) -> list[float]:
        grayscale = skimage.color.rgb2gray(image[..., :3], channel_axis = -1)
        return [float(np.mean(grayscale))]
