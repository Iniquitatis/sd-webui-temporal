import scipy
import skimage

from temporal.general_data import GeneralData
from temporal.object import Field
from temporal.pipeline_modules.filtering import ImageFilter
from temporal.utils.image import NumpyImage, apply_channelwise


class MedianFilter(ImageFilter):
    name = "Median"

    radius: int = Field(0, name = "Radius", minimum = 0, maximum = 50, step = 1, display = "slider")
    percentile: float = Field(50.0, name = "Percentile", minimum = 0.0, maximum = 100.0, step = 0.1, suffix = "%", display = "slider")

    def process(self, image: NumpyImage, general: GeneralData, iter_index: int, seed: int) -> NumpyImage:
        footprint = skimage.morphology.disk(self.radius)

        if self.percentile == 50.0:
            filter = lambda x: scipy.ndimage.median_filter(x, footprint = footprint, mode = "nearest")
        else:
            filter = lambda x: scipy.ndimage.percentile_filter(x, self.percentile, footprint = footprint, mode = "nearest")

        return apply_channelwise(image, filter)
