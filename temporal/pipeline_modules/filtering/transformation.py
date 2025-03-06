import skimage

from temporal.general_data import GeneralData
from temporal.meta.configurable import ConfigurableParam as Param
from temporal.pipeline_modules.filtering import ImageFilter
from temporal.utils.image import NumpyImage, make_trs_transform
from temporal.vector import FloatVector


class TransformationFilter(ImageFilter):
    name = "Transformation"

    translation: FloatVector = Param("Translation", axes = ["X", "Y"], minimum = -1.0, maximum = 1.0, step = 0.001, value = lambda: FloatVector(0.0, 0.0), ui_type = "slider")
    rotation: float = Param("Rotation", minimum = -90.0, maximum = 90.0, step = 0.1, value = 0.0, ui_type = "slider")
    scale: float = Param("Scale", minimum = 0.0, maximum = 2.0, step = 0.001, value = 1.0, ui_type = "slider")

    def process(self, image: NumpyImage, general: GeneralData, frame_index: int, seed: int) -> NumpyImage:
        return skimage.transform.warp(image, make_trs_transform(
            image_size = (image.shape[1], image.shape[0]),
            translation = (self.translation.x, self.translation.y),
            rotation = self.rotation,
            scale = self.scale,
        ), mode = "symmetric")
