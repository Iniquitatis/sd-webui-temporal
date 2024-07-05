import numpy as np
import skimage

from temporal.pipeline_modules.measuring import MeasuringModule
from temporal.utils.image import NumpyImage


class LuminanceSigmaMeasuringModule(MeasuringModule):
    id = "luminance_sigma_measuring"
    name = "Luminance sigma"
    file_name = "luminance_sigma"
    channels = [
        ("Luminance", "gray"),
    ]

    def measure(self, npim: NumpyImage) -> list[float]:
        grayscale = skimage.color.rgb2gray(npim[..., :3], channel_axis = -1)
        return [float(np.std(grayscale))]
