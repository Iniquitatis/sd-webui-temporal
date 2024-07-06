import skimage

from temporal.pipeline_modules.measuring import MeasuringModule
from temporal.utils.image import NumpyImage


class NoiseSigmaMeasuringModule(MeasuringModule):
    name = "Noise sigma"
    file_name = "noise_sigma"
    channels = [
        ("Noise sigma", "royalblue"),
    ]

    def measure(self, npim: NumpyImage) -> list[float]:
        return [float(skimage.restoration.estimate_sigma(npim, average_sigmas = True, channel_axis = -1))]
