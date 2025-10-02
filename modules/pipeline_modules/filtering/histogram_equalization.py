import skimage

from modules.general_data import GeneralData
from modules.pipeline_modules.filtering import ImageFilter
from modules.utils.image import NumpyImage


class HistogramEqualizationFilter(ImageFilter):
    name = "Histogram equalization"

    def process(self, image: NumpyImage, general: GeneralData) -> NumpyImage:
        return skimage.exposure.equalize_hist(image)
