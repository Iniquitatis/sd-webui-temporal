from abc import abstractmethod
from typing import Optional

import skimage

from temporal.blend_modes import BLEND_MODES
from temporal.image_mask import ImageMask
from temporal.meta.serializable import SerializableField as Field
from temporal.pipeline_module import PipelineModule
from temporal.project import Project
from temporal.utils.image import NumpyImage, match_image
from temporal.utils.math import lerp, normalize
from temporal.utils.numpy import saturate_array


class ImageFilter(PipelineModule, abstract = True):
    icon = "\U00002728"

    amount: float = Field(1.0)
    amount_relative: bool = Field(False)
    blend_mode: str = Field("normal")
    mask: ImageMask = Field(factory = ImageMask)

    def forward(self, images: list[NumpyImage], project: Project, frame_index: int, seed: int) -> Optional[list[NumpyImage]]:
        return [saturate_array(self._blend(x, self.process(x, i, project, frame_index, seed + i), project)) for i, x in enumerate(images)]

    @abstractmethod
    def process(self, npim: NumpyImage, parallel_index: int, project: Project, frame_index: int, seed: int) -> NumpyImage:
        raise NotImplementedError

    def _blend(self, npim: NumpyImage, processed: NumpyImage, project: Project) -> NumpyImage:
        if npim is processed:
            return npim

        amount = self.amount * (project.processing.denoising_strength if self.amount_relative else 1.0)

        if amount == 0.0:
            return npim

        processed = BLEND_MODES[self.blend_mode].blend(npim, processed)

        if amount == 1.0 and self.mask.image is None:
            return processed

        if self.mask.image is not None:
            factor = match_image(self.mask.image, npim)

            if self.mask.normalized:
                factor = normalize(factor, factor.min(), factor.max())

            if self.mask.inverted:
                factor = 1.0 - factor

            if self.mask.blurring:
                factor = skimage.filters.gaussian(factor, round(self.mask.blurring), channel_axis = -1)

        else:
            factor = 1.0

        factor *= amount

        return lerp(npim, processed, factor)
