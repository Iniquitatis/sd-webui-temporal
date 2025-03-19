import skimage

from temporal.general_data import GeneralData
from temporal.object import Param
from temporal.pipeline_modules.filtering import ImageFilter
from temporal.utils.image import NumpyImage, apply_channelwise


class MorphologyFilter(ImageFilter):
    name = "Morphology"

    mode: str = Param("Mode", choices = {
        "erosion": "Erosion",
        "dilation": "Dilation",
        "opening": "Opening",
        "closing": "Closing",
    }, value = "erosion", ui_type = "radio")
    radius: int = Param("Radius", minimum = 0, maximum = 50, step = 1, value = 0, ui_type = "slider")

    def process(self, image: NumpyImage, general: GeneralData, iter_index: int, seed: int) -> NumpyImage:
        func = (
            skimage.morphology.erosion  if self.mode == "erosion"  else
            skimage.morphology.dilation if self.mode == "dilation" else
            skimage.morphology.opening  if self.mode == "opening"  else
            skimage.morphology.closing  if self.mode == "closing"  else
            lambda image, footprint: image
        )
        footprint = skimage.morphology.disk(self.radius)
        return apply_channelwise(image, lambda x: func(x, footprint))
