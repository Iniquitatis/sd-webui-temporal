import skimage

from modules.general_data import GeneralData
from modules.pipeline_modules.filtering import ImageFilter
from modules.utils.image import NumpyImage


class ContrastNormalizationFilter(ImageFilter):
    name = "Contrast normalization"

    def process(self, image: NumpyImage, general: GeneralData) -> NumpyImage:
        return skimage.exposure.rescale_intensity(image)
