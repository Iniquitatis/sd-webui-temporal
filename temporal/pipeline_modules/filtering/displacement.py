import numpy as np
import skimage

from temporal.general_data import GeneralData
from temporal.image_source import ImageSource
from temporal.meta.configurable import FloatVectorParam, ImageSourceParam
from temporal.pipeline_modules.filtering import ImageFilter
from temporal.utils.image import NumpyImage, apply_channelwise, ensure_image_dims
from temporal.vector import FloatVector


class DisplacementFilter(ImageFilter):
    name = "Displacement"

    source: ImageSource = ImageSourceParam("Image source", channels = 3)
    scale: FloatVector = FloatVectorParam("Scale", axes = ["X", "Y"], step = 0.1, factory = lambda: FloatVector(1.0, 1.0), ui_type = "box")

    def process(self, npim: NumpyImage, parallel_index: int, general: GeneralData, frame_index: int, seed: int) -> NumpyImage:
        if (image := self.source.get_image(general.image, frame_index - 1)) is None:
            return npim

        image = ensure_image_dims(image, size = (general.image_size.x, general.image_size.y))

        gradient = image[..., :2] * 2.0 - 1.0

        coords = np.indices(npim.shape[:2]).astype(np.float64)
        coords[[1, 0], ...] += (gradient * self.scale.to_numpy()).transpose(2, 0, 1)

        return apply_channelwise(npim, lambda x: skimage.transform.warp(x, coords, mode = "symmetric"))
