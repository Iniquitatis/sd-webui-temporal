from functools import lru_cache
from typing import Optional

import skimage

from temporal.general_data import GeneralData
from temporal.object import Field, Meta, Object, Static
from temporal.shared import shared
from temporal.utils import logging
from temporal.utils.image import NumpyImage, ensure_image_dims, make_trs_transform
from temporal.vector import IntVector


class PipelineModule(Object, abstract = True):
    name: str = Meta("UNDEFINED")
    is_filter: bool = Meta(False)
    is_sampleable: bool = Meta(False)
    sample_iterations: int = Static(1)

    enabled: bool = Field(True)
    preview: bool = Field(True)

    def forward(self, image: NumpyImage, general: GeneralData, frame_index: int, seed: int) -> Optional[NumpyImage]:
        return image

    def finalize(self, image: NumpyImage, general: GeneralData) -> None:
        pass

    def reset(self, general: GeneralData) -> None:
        pass

    def sample(self, size: tuple[int, int]) -> NumpyImage:
        sample_image = _get_scaled_sample_image(size)
        last_image = sample_image.copy()

        for i in range(self.sample_iterations):
            if (result := self.forward(last_image, GeneralData(
                initial_image = sample_image,
                seed = 31337,
            ), i + 1, 31337 + i)) is not None:
                last_image = result
            else:
                logging.warning("Module couldn't render an image for some reason")
                return sample_image

            if (i + 1) != self.sample_iterations:
                last_image = skimage.transform.warp(last_image, make_trs_transform(
                    image_size = size,
                    translation = (0.01, 0.01),
                    rotation = 3.0,
                    scale = 0.99,
                ), mode = "symmetric")

        return last_image


@lru_cache
def _get_scaled_sample_image(size: tuple[int, int]) -> NumpyImage:
    return ensure_image_dims(shared.sample_image, size, 3)
