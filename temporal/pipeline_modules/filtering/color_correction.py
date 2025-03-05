import skimage

from temporal.general_data import GeneralData
from temporal.image_source import ImageSource
from temporal.meta.configurable import ConfigurableParam as Param
from temporal.pipeline_modules.filtering import ImageFilter
from temporal.utils.image import NumpyImage, match_image


class ColorCorrectionFilter(ImageFilter):
    name = "Color correction"

    source: ImageSource = Param("Image source", channels = 3, value = ImageSource)
    normalize_contrast: bool = Param("Normalize contrast", value = False)
    equalize_histogram: bool = Param("Equalize histogram", value = False)

    def process(self, npim: NumpyImage, general: GeneralData, frame_index: int, seed: int) -> NumpyImage:
        if (image := self.source.get_image(general.initial_image, frame_index - 1)) is not None:
            npim = skimage.exposure.match_histograms(npim, match_image(image, npim, size = False), channel_axis = -1)

        if self.normalize_contrast:
            npim = skimage.exposure.rescale_intensity(npim)

        if self.equalize_histogram:
            npim = skimage.exposure.equalize_hist(npim)

        return npim
