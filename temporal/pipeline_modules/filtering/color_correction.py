import skimage

from temporal.general_data import GeneralData
from temporal.image_source import ImageSource
from temporal.object import Param
from temporal.pipeline_modules.filtering import ImageFilter
from temporal.utils.image import NumpyImage, match_image


class ColorCorrectionFilter(ImageFilter):
    name = "Color correction"

    source: ImageSource = Param("Image source", channels = 3, value = ImageSource)
    normalize_contrast: bool = Param("Normalize contrast", value = False)
    equalize_histogram: bool = Param("Equalize histogram", value = False)

    def process(self, image: NumpyImage, general: GeneralData, frame_index: int, seed: int) -> NumpyImage:
        if (source := self.source.get_image(general.initial_image, frame_index - 1)) is not None:
            image = skimage.exposure.match_histograms(image, match_image(source, image, size = False), channel_axis = -1)

        if self.normalize_contrast:
            image = skimage.exposure.rescale_intensity(image)

        if self.equalize_histogram:
            image = skimage.exposure.equalize_hist(image)

        return image
