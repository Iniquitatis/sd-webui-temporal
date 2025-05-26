import skimage

from temporal.general_data import GeneralData
from temporal.object import Field
from temporal.pipeline_modules.filtering import ImageFilter
from temporal.utils.image import NumpyImage, apply_channelwise


class MorphologyFilter(ImageFilter):
    name = "Morphology"

    mode: str = Field("erosion", name = "Mode", choices = {
        "erosion": "Erosion",
        "dilation": "Dilation",
        "opening": "Opening",
        "closing": "Closing",
    }, display = "radio")
    radius: int = Field(0, name = "Radius", minimum = 0, maximum = 50, step = 1, display = "slider")

    def process(self, image: NumpyImage, general: GeneralData, iter_index: int, seed: int) -> NumpyImage:
        func = _MODES[self.mode]
        footprint = skimage.morphology.disk(self.radius)
        return apply_channelwise(image, lambda x: func(x, footprint))


_MODES = {
    "erosion": skimage.morphology.erosion,
    "dilation": skimage.morphology.dilation,
    "opening": skimage.morphology.opening,
    "closing": skimage.morphology.closing,
}
