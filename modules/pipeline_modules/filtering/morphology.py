import skimage

from modules.general_data import GeneralData
from modules.object import Field
from modules.pipeline_modules.filtering import ImageFilter
from modules.utils.image import NumpyImage, apply_channelwise


class MorphologyFilter(ImageFilter):
    name = "Morphology"

    mode: str = Field("erosion", name = "Mode", choices = {
        "erosion": "Erosion",
        "dilation": "Dilation",
        "opening": "Opening",
        "closing": "Closing",
    }, display = "radio")
    radius: int = Field(0, name = "Radius", minimum = 0, maximum = 50, step = 1, display = "slider")

    def process(self, image: NumpyImage, general: GeneralData) -> NumpyImage:
        func = _MODES[self.mode]
        footprint = skimage.morphology.disk(self.radius)
        return apply_channelwise(image, lambda x: func(x, footprint))


_MODES = {
    "erosion": skimage.morphology.erosion,
    "dilation": skimage.morphology.dilation,
    "opening": skimage.morphology.opening,
    "closing": skimage.morphology.closing,
}
