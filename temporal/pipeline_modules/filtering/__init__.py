from abc import abstractmethod

from temporal.blend_modes import BlendMode, NormalBlendMode
from temporal.general_data import GeneralData
from temporal.image_mask import ImageMask
from temporal.object import Field
from temporal.pipeline_module import PipelineModule
from temporal.pipeline_state import PipelineResult, PipelineState
from temporal.utils.image import NumpyImage
from temporal.utils.math import lerp
from temporal.utils.numpy import saturate_array


class ImageFilter(PipelineModule, abstract = True):
    is_filter = True
    is_sampleable = True

    amount: float = Field(1.0, name = "Amount", minimum = 0.0, maximum = 1.0, step = 0.01, display = "slider")
    blend_mode: BlendMode = Field(NormalBlendMode, name = "Blend mode")
    mask: ImageMask = Field(ImageMask, name = "Mask", display = "accordion")

    def forward(self, image: NumpyImage, general: GeneralData) -> PipelineResult:
        yield PipelineState.finish(image = self._blend(image, self.process(image, general)), preview = self.preview)

    @abstractmethod
    def process(self, image: NumpyImage, general: GeneralData) -> NumpyImage:
        raise NotImplementedError

    def _blend(self, image: NumpyImage, processed: NumpyImage) -> NumpyImage:
        if self.amount == 0.0:
            return image

        result = image.copy()
        result[..., :3] = saturate_array(self.blend_mode.blend(image[..., :3], processed[..., :3]))
        result[:] = self.mask.mask(image, result)
        result[:] = lerp(image, result, self.amount)

        return result
