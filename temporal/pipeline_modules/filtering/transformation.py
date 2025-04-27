import skimage

from temporal.general_data import GeneralData
from temporal.object import Field
from temporal.pipeline_modules.filtering import ImageFilter
from temporal.utils.image import NumpyImage, make_trs_transform
from temporal.vector import FloatVector


class TransformationFilter(ImageFilter):
    name = "Transformation"

    translation: FloatVector = Field(lambda: FloatVector(0.0, 0.0), name = "Translation", axes = ["X", "Y"], minimum = -1.0, maximum = 1.0, step = 0.001, display = "slider")
    rotation: float = Field(0.0, name = "Rotation", minimum = -90.0, maximum = 90.0, step = 0.1, display = "slider")
    scale: float = Field(1.0, name = "Scale", minimum = 0.0, maximum = 2.0, step = 0.001, display = "slider")

    def process(self, image: NumpyImage, general: GeneralData, iter_index: int, seed: int) -> NumpyImage:
        return skimage.transform.warp(image, make_trs_transform(
            image_size = (image.shape[1], image.shape[0]),
            translation = (self.translation.x, self.translation.y),
            rotation = self.rotation,
            scale = self.scale,
        ), mode = "symmetric")
