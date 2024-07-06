import numpy as np
import skimage

from temporal.meta.configurable import FloatParam
from temporal.pipeline_modules.filtering import ImageFilter
from temporal.project import Project
from temporal.utils.image import NumpyImage


class TransformationFilter(ImageFilter):
    name = "Transformation"

    translation_x: float = FloatParam("Translation X", minimum = -1.0, maximum = 1.0, step = 0.001, value = 0.0, ui_type = "slider")
    translation_y: float = FloatParam("Translation Y", minimum = -1.0, maximum = 1.0, step = 0.001, value = 0.0, ui_type = "slider")
    rotation: float = FloatParam("Rotation", minimum = -90.0, maximum = 90.0, step = 0.1, value = 0.0, ui_type = "slider")
    scaling: float = FloatParam("Scaling", minimum = 0.0, maximum = 2.0, step = 0.001, value = 1.0, ui_type = "slider")

    def process(self, npim: NumpyImage, parallel_index: int, project: Project, frame_index: int, seed: int) -> NumpyImage:
        height, width = npim.shape[:2]

        o_transform = skimage.transform.AffineTransform(translation = (-width / 2, -height / 2))
        t_transform = skimage.transform.AffineTransform(translation = (-self.translation_x * width, -self.translation_y * height))
        r_transform = skimage.transform.AffineTransform(rotation = np.deg2rad(self.rotation))
        s_transform = skimage.transform.AffineTransform(scale = self.scaling)

        return skimage.transform.warp(npim, skimage.transform.AffineTransform(t_transform.params @ np.linalg.inv(o_transform.params) @ s_transform.params @ r_transform.params @ o_transform.params).inverse, mode = "symmetric")
