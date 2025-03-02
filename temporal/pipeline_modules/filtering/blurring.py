import skimage

from temporal.general_data import GeneralData
from temporal.meta.configurable import ConfigurableParam as Param
from temporal.pipeline_modules.filtering import ImageFilter
from temporal.utils.image import NumpyImage


class BlurringFilter(ImageFilter):
    name = "Blurring"

    radius: float = Param("Radius", minimum = 0.0, maximum = 50.0, step = 0.1, value = 0.0, ui_type = "slider")

    def process(self, npim: NumpyImage, general: GeneralData, frame_index: int, seed: int) -> NumpyImage:
        return skimage.filters.gaussian(npim, round(self.radius), channel_axis = -1)
