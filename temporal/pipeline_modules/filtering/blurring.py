import skimage

from temporal.meta.configurable import FloatParam
from temporal.pipeline_modules.filtering import ImageFilter
from temporal.project import Project
from temporal.utils.image import NumpyImage


class BlurringFilter(ImageFilter):
    id = "blurring"
    name = "Blurring"

    radius: float = FloatParam("Radius", minimum = 0.0, maximum = 50.0, step = 0.1, value = 0.0, ui_type = "slider")

    def process(self, npim: NumpyImage, parallel_index: int, project: Project, frame_index: int, seed: int) -> NumpyImage:
        return skimage.filters.gaussian(npim, round(self.radius), channel_axis = -1)
