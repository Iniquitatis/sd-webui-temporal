from abc import abstractmethod
from typing import Optional

from temporal.blend_modes import BlendMode, NormalBlendMode
from temporal.general_data import GeneralData
from temporal.image_mask import ImageMask
from temporal.meta.serializable import SerializableField as Field
from temporal.pipeline_module import PipelineModule
from temporal.utils.image import NumpyImage
from temporal.utils.math import lerp
from temporal.utils.numpy import saturate_array


class ImageFilter(PipelineModule, abstract = True):
    is_sampleable = True

    amount: float = Field(1.0)
    blend_mode: BlendMode = Field(NormalBlendMode)
    mask: ImageMask = Field(ImageMask)

    def forward(self, image: NumpyImage, general: GeneralData, frame_index: int, seed: int) -> Optional[NumpyImage]:
        return self._blend(image, self.process(image, general, frame_index, seed))

    @abstractmethod
    def process(self, npim: NumpyImage, general: GeneralData, frame_index: int, seed: int) -> NumpyImage:
        raise NotImplementedError

    def _blend(self, npim: NumpyImage, processed: NumpyImage) -> NumpyImage:
        if self.amount == 0.0:
            return npim

        processed = self.blend_mode.blend(npim, processed)
        processed = self.mask.mask(npim, processed)

        return saturate_array(lerp(npim, processed, self.amount))
