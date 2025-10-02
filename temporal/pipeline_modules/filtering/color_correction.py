import skimage

from temporal.general_data import GeneralData
from temporal.image_source import ImageSource
from temporal.object import Field
from temporal.pipeline_modules.filtering import ImageFilter
from temporal.utils.image import NumpyImage, match_image


class ColorCorrectionFilter(ImageFilter):
    name = "Color correction"

    source: ImageSource = Field(ImageSource, name = "Image source", channels = 3, display = "unpack")
    iteration: int = Field(0, flags = {"runtime"})

    def process(self, image: NumpyImage, general: GeneralData) -> NumpyImage:
        if (source := self.source.get_image(general.initial_image, self.iteration)) is not None:
            result = skimage.exposure.match_histograms(image, match_image(source, image, size = False), channel_axis = -1)
        else:
            result = image

        self.iteration += 1

        return result
