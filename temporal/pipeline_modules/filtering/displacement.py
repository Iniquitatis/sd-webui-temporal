import numpy as np
import skimage

from temporal.general_data import GeneralData
from temporal.image_source import ImageSource
from temporal.object import Param
from temporal.pipeline_modules.filtering import ImageFilter
from temporal.utils.image import NumpyImage, apply_channelwise, ensure_image_dims
from temporal.utils.numpy import FloatType
from temporal.vector import FloatVector


class DisplacementFilter(ImageFilter):
    name = "Displacement"

    source: ImageSource = Param("Image source", channels = 3, value = ImageSource)
    scale: FloatVector = Param("Scale", axes = ["X", "Y"], step = 0.1, value = lambda: FloatVector(1.0, 1.0), ui_type = "box")

    def process(self, image: NumpyImage, general: GeneralData, iter_index: int, seed: int) -> NumpyImage:
        if (source := self.source.get_image(general.initial_image, iter_index - 1)) is None:
            return image

        source = ensure_image_dims(source, size = (general.image_size.x, general.image_size.y))

        gradient = source[..., :2] * 2.0 - 1.0

        coords = np.indices(image.shape[:2]).astype(FloatType)
        coords[[1, 0], ...] += (gradient * self.scale.to_numpy()).transpose(2, 0, 1)

        return apply_channelwise(image, lambda x: skimage.transform.warp(x, coords, mode = "symmetric"))
