import skimage

from modules.pipeline_modules.measuring import MeasuringModule
from modules.utils.image import NumpyImage


class NoiseSigmaMeasuringModule(MeasuringModule):
    name = "Noise sigma"
    file_name = "noise_sigma"
    channels = [
        ("Noise sigma", "royalblue"),
    ]

    def measure(self, image: NumpyImage) -> list[float]:
        return [float(skimage.restoration.estimate_sigma(image, average_sigmas = True, channel_axis = -1))]
