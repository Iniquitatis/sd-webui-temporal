import skimage

from temporal.general_data import GeneralData
from temporal.pipeline_modules.filtering import ImageFilter
from temporal.utils.image import NumpyImage


class HistogramEqualizationFilter(ImageFilter):
    name = "Histogram equalization"

    def process(self, image: NumpyImage, general: GeneralData, iter_index: int, seed: int) -> NumpyImage:
        return skimage.exposure.equalize_hist(image)
