import numpy as np
import skimage

from temporal.general_data import GeneralData
from temporal.image_source import ImageSource
from temporal.object import Field
from temporal.pipeline_modules.painting import PaintingModule
from temporal.utils.image import NumpyImage, ensure_image_dims, make_trs_transform
from temporal.utils.numpy import saturate_array
from temporal.vector import FloatVector


class ImagePaintingModule(PaintingModule):
    name = "Image"

    source: ImageSource = Field(ImageSource, name = "Image source", channels = 4, display = "group")
    offset: FloatVector = Field(lambda: FloatVector(0.0, 0.0), name = "Offset", axes = ["X", "Y"], minimum = -1.0, maximum = 1.0, step = 0.001, display = "slider")
    blurring: float = Field(0.0, name = "Blurring", minimum = 0.0, maximum = 50.0, step = 0.1, display = "slider")
    iteration: int = Field(0, flags = {"runtime"})

    def draw(self, size: tuple[int, int], general: GeneralData) -> NumpyImage:
        if (image := self.source.get_image(general.initial_image, self.iteration)) is None:
            return np.zeros((size[1], size[0], 4))

        self.iteration += 1

        return skimage.transform.warp(
            saturate_array(skimage.filters.gaussian(ensure_image_dims(image, size = size), round(self.blurring), channel_axis = -1)),
            make_trs_transform(size, translation = (self.offset.x, self.offset.y)),
        )
