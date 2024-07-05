import numpy as np
import skimage

from temporal.pipeline_modules.measuring import MeasuringModule
from temporal.utils.image import NumpyImage


class LuminanceMeanMeasuringModule(MeasuringModule):
    id = "luminance_mean_measuring"
    name = "Luminance mean"
    file_name = "luminance_mean"
    channels = [
        ("Luminance", "gray"),
    ]

    def measure(self, npim: NumpyImage) -> list[float]:
        grayscale = skimage.color.rgb2gray(npim[..., :3], channel_axis = -1)
        return [float(np.mean(grayscale))]
