from functools import lru_cache
from typing import Literal, Optional

import skimage

from temporal.general_data import GeneralData
from temporal.object import Field, Object, Static
from temporal.shared import shared
from temporal.utils import logging
from temporal.utils.image import NumpyImage, PILImage, ensure_image_dims, make_trs_transform
from temporal.video import Video


VisualizableType = NumpyImage | PILImage | Video


class PipelineModule(Object, abstract = True):
    name: str = Static("UNDEFINED")
    is_filter: bool = Static(False)
    is_visualizable: bool = Static(False)
    visualization_type: Literal["image", "video"] = Static("image")
    is_sampleable: bool = Static(False)
    sample_iterations: int = Static(1, flags = {"private"})

    enabled: bool = Field(True, name = "Enabled")
    preview: bool = Field(True, name = "Preview")

    def forward(self, image: NumpyImage, general: GeneralData, iter_index: int, seed: int) -> Optional[NumpyImage]:
        return image

    def finalize(self, image: NumpyImage, general: GeneralData) -> None:
        pass

    def reset(self, general: GeneralData) -> None:
        pass

    def visualize(self, general: GeneralData) -> VisualizableType:
        raise NotImplementedError

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
