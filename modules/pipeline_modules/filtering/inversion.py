from modules.general_data import GeneralData
from modules.pipeline_modules.filtering import ImageFilter
from modules.utils.image import NumpyImage


class InversionFilter(ImageFilter):
    name = "Inversion"

    def process(self, image: NumpyImage, general: GeneralData) -> NumpyImage:
        return 1.0 - image
