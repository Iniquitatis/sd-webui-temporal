import skimage

from temporal.meta.configurable import FloatParam
from temporal.pipeline_modules.filtering import ImageFilter
from temporal.project import Project
from temporal.utils.image import NumpyImage


class SharpeningFilter(ImageFilter):
    id = "sharpening"
    name = "Sharpening"

    strength: float = FloatParam("Strength", minimum = 0.0, maximum = 1.0, step = 0.01, value = 0.0, ui_type = "slider")
    radius: float = FloatParam("Radius", minimum = 0.0, maximum = 5.0, step = 0.1, value = 0.0, ui_type = "slider")

    def process(self, npim: NumpyImage, parallel_index: int, project: Project, frame_index: int, seed: int) -> NumpyImage:
        # NOTE: `ndim - 1` is intentional, as there's probably a bug in skimage
        return skimage.filters.unsharp_mask(npim, self.radius, self.strength, channel_axis = npim.ndim - 1)
