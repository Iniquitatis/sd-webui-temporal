import numpy as np
import skimage

from temporal.image_source import ImageSource
from temporal.meta.configurable import FloatParam, ImageSourceParam
from temporal.pipeline_modules.painting import PaintingModule
from temporal.project import Project
from temporal.utils.image import NumpyImage, ensure_image_dims


class ImagePaintingModule(PaintingModule):
    name = "Image"

    source: ImageSource = ImageSourceParam("Image source", channels = 4)
    blurring: float = FloatParam("Blurring", minimum = 0.0, maximum = 50.0, step = 0.1, value = 0.0, ui_type = "slider")

    def draw(self, size: tuple[int, int], parallel_index: int, project: Project, frame_index: int, seed: int) -> NumpyImage:
        if (image := self.source.get_image(project.backend_data.images[parallel_index], frame_index - 1)) is None:
            return np.zeros((size[1], size[0], 4))

        return ensure_image_dims(skimage.filters.gaussian(image, round(self.blurring), channel_axis = -1), size = size)
