from abc import abstractmethod
from typing import Optional

from temporal.blend_modes import BlendMode, NormalBlendMode
from temporal.image_mask import ImageMask
from temporal.meta.serializable import SerializableField as Field
from temporal.pipeline_module import PipelineModule
from temporal.project import Project
from temporal.utils.image import NumpyImage
from temporal.utils.math import lerp
from temporal.utils.numpy import saturate_array


class ImageFilter(PipelineModule, abstract = True):
    icon = "\U00002728"

    amount: float = Field(1.0)
    amount_relative: bool = Field(False)
    blend_mode: BlendMode = Field(factory = NormalBlendMode)
    mask: ImageMask = Field(factory = ImageMask)

    def forward(self, images: list[NumpyImage], project: Project, frame_index: int, seed: int) -> Optional[list[NumpyImage]]:
        return [saturate_array(self._blend(x, self.process(x, i, project, frame_index, seed + i), project)) for i, x in enumerate(images)]

    @abstractmethod
    def process(self, npim: NumpyImage, parallel_index: int, project: Project, frame_index: int, seed: int) -> NumpyImage:
        raise NotImplementedError

    def _blend(self, npim: NumpyImage, processed: NumpyImage, project: Project) -> NumpyImage:
        amount = self.amount * (project.backend_data.denoising_strength if self.amount_relative else 1.0)

        if amount == 0.0:
            return npim

        processed = self.blend_mode.blend(npim, processed)
        processed = self.mask.mask(npim, processed)

        return lerp(npim, processed, amount)
