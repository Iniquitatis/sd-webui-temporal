from functools import lru_cache
from typing import Any, Optional, Type
from uuid import uuid4

from temporal.general_data import GeneralData
from temporal.meta.configurable import Configurable
from temporal.meta.serializable import SerializableField as Field
from temporal.shared import shared
from temporal.utils.image import NumpyImage, ensure_image_dims
from temporal.utils.logging import warning
from temporal.vector import IntVector


PIPELINE_MODULES: list[Type["PipelineModule"]] = []


class PipelineModule(Configurable, abstract = True):
    store = PIPELINE_MODULES

    is_sampleable: bool = False

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

        if (result := self.forward(sample_image, GeneralData(
            initial_image = sample_image,
            image_size = IntVector(*size),
            seed = 31337,
        ), 1, 31337)) is not None:
            return result
        else:
            warning("Module couldn't render an image for some reason")
            return sample_image


@lru_cache
def _get_scaled_sample_image(size: tuple[int, int]) -> NumpyImage:
    return ensure_image_dims(shared.sample_image, size, 3)
