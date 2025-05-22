import skimage

from temporal.general_data import GeneralData
from temporal.pipeline_modules.filtering import ImageFilter
from temporal.utils.image import NumpyImage


class ContrastNormalizationFilter(ImageFilter):
    name = "Contrast normalization"

    def process(self, image: NumpyImage, general: GeneralData, iter_index: int, seed: int) -> NumpyImage:
        return skimage.exposure.rescale_intensity(image)
