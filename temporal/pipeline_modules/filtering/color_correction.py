import skimage

from temporal.image_source import ImageSource
from temporal.meta.configurable import BoolParam, ImageSourceParam
from temporal.pipeline_modules.filtering import ImageFilter
from temporal.project import Project
from temporal.utils.image import NumpyImage, match_image


class ColorCorrectionFilter(ImageFilter):
    id = "color_correction"
    name = "Color correction"

    source: ImageSource = ImageSourceParam("Image source", channels = 3)
    normalize_contrast: bool = BoolParam("Normalize contrast", value = False)
    equalize_histogram: bool = BoolParam("Equalize histogram", value = False)

    def process(self, npim: NumpyImage, parallel_index: int, project: Project, frame_index: int, seed: int) -> NumpyImage:
        if (image := self.source.get_image(project.processing.init_images[parallel_index], frame_index - 1)) is not None:
            npim = skimage.exposure.match_histograms(npim, match_image(image, npim, size = False), channel_axis = -1)

        if self.normalize_contrast:
            npim = skimage.exposure.rescale_intensity(npim)

        if self.equalize_histogram:
            npim = skimage.exposure.equalize_hist(npim)

        return npim
