import numpy as np
import skimage

from temporal.image_source import ImageSource
from temporal.meta.configurable import FloatParam, ImageSourceParam
from temporal.pipeline_modules.filtering import ImageFilter
from temporal.project import Project
from temporal.utils.image import NumpyImage, apply_channelwise, ensure_image_dims


class DisplacementFilter(ImageFilter):
    name = "Displacement"

    source: ImageSource = ImageSourceParam("Image source", channels = 3)
    x_scale: float = FloatParam("X scale", step = 0.1, value = 1.0, ui_type = "box")
    y_scale: float = FloatParam("Y scale", step = 0.1, value = 1.0, ui_type = "box")

    def process(self, npim: NumpyImage, parallel_index: int, project: Project, frame_index: int, seed: int) -> NumpyImage:
        if (image := self.source.get_image(project.backend_data.images[parallel_index], frame_index - 1)) is None:
            return npim

        image = ensure_image_dims(image, size = (project.backend_data.width, project.backend_data.height))

        gradient = image[..., :2] * 2.0 - 1.0

        height, width = npim.shape[:2]

        coords = np.indices((height, width)).astype(np.float_)
        coords[0] += gradient[..., 1] * self.y_scale
        coords[1] += gradient[..., 0] * self.x_scale

        return apply_channelwise(npim, lambda x: skimage.transform.warp(x, coords, mode = "symmetric"))
