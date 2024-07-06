import scipy
import skimage

from temporal.meta.configurable import FloatParam, IntParam
from temporal.pipeline_modules.filtering import ImageFilter
from temporal.project import Project
from temporal.utils.image import NumpyImage, apply_channelwise


class MedianFilter(ImageFilter):
    name = "Median"

    radius: int = IntParam("Radius", minimum = 0, maximum = 50, step = 1, value = 0, ui_type = "slider")
    percentile: float = FloatParam("Percentile", minimum = 0.0, maximum = 100.0, step = 0.1, value = 50.0, ui_type = "slider")

    def process(self, npim: NumpyImage, parallel_index: int, project: Project, frame_index: int, seed: int) -> NumpyImage:
        footprint = skimage.morphology.disk(self.radius)

        if self.percentile == 50.0:
            filter = lambda x: scipy.ndimage.median_filter(x, footprint = footprint, mode = "nearest")
        else:
            filter = lambda x: scipy.ndimage.percentile_filter(x, self.percentile, footprint = footprint, mode = "nearest")

        return apply_channelwise(npim, filter)
