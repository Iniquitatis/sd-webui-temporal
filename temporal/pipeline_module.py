from functools import lru_cache
from typing import Any, Optional, Type
from uuid import uuid4

import skimage

from temporal.general_data import GeneralData
from temporal.meta.configurable import Configurable
from temporal.meta.serializable import SerializableField as Field
from temporal.shared import shared
from temporal.utils.image import NumpyImage, ensure_image_dims, make_trs_transform
from temporal.utils.logging import warning
from temporal.vector import IntVector


PIPELINE_MODULES: list[Type["PipelineModule"]] = []


class PipelineModule(Configurable, abstract = True):
    store = PIPELINE_MODULES

    is_sampleable: bool = False
    sample_iterations: int = 1

    uuid: str = Field("")
    enabled: bool = Field(True)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        if not self.uuid:
            self.uuid = str(uuid4())

    @classmethod
    def schema(cls) -> dict[str, Any]:
        from temporal.pipeline_modules.filtering import ImageFilter

        return {
            **super().schema(),
            "is_filter": issubclass(cls, ImageFilter),
            "is_sampleable": cls.is_sampleable,
        }

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
                image_size = IntVector(*size),
                seed = 31337,
            ), i + 1, 31337 + i)) is not None:
                last_image = result
            else:
                warning("Module couldn't render an image for some reason")
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
