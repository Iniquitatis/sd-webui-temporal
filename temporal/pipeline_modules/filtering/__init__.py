from abc import abstractmethod
from typing import Optional

from temporal.blend_modes import BlendMode, NormalBlendMode
from temporal.general_data import GeneralData
from temporal.image_mask import ImageMask
from temporal.object import Field
from temporal.pipeline_module import PipelineModule
from temporal.utils.image import NumpyImage
from temporal.utils.math import lerp
from temporal.utils.numpy import saturate_array


class ImageFilter(PipelineModule, abstract = True):
    is_filter = True
    is_sampleable = True

    amount: float = Field(1.0, name = "Amount", minimum = 0.0, maximum = 1.0, step = 0.01, display = "slider")
    blend_mode: BlendMode = Field(NormalBlendMode, name = "Blend mode")
    mask: ImageMask = Field(ImageMask, name = "Image mask", display = "accordion")

    def forward(self, image: NumpyImage, general: GeneralData, iter_index: int, seed: int) -> Optional[NumpyImage]:
        return self._blend(image, self.process(image, general, iter_index, seed))

    @abstractmethod
    def process(self, image: NumpyImage, general: GeneralData, iter_index: int, seed: int) -> NumpyImage:
        raise NotImplementedError

    def _blend(self, image: NumpyImage, processed: NumpyImage) -> NumpyImage:
        if self.amount == 0.0:
            return image

        processed = saturate_array(self.blend_mode.blend(image, processed))
        processed = self.mask.mask(image, processed)

        return lerp(image, processed, self.amount)
