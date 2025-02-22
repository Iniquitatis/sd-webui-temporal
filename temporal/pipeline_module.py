from typing import Any, Optional, Type

from temporal.general_data import GeneralData
from temporal.meta.configurable import Configurable
from temporal.meta.serializable import SerializableField as Field
from temporal.utils.image import NumpyImage


PIPELINE_MODULES: list[Type["PipelineModule"]] = []


class PipelineModule(Configurable, abstract = True):
    store = PIPELINE_MODULES

    icon: str = "\U00002699\ufe0f"

    enabled: bool = Field(True)

    @classmethod
    def schema(cls) -> dict[str, Any]:
        from temporal.pipeline_modules.filtering import ImageFilter

        return super().schema() | {
            "icon": cls.icon,
            "is_filter": issubclass(cls, ImageFilter),
        }

    def forward(self, images: list[NumpyImage], general: GeneralData, frame_index: int, seed: int) -> Optional[list[NumpyImage]]:
        return images

    def finalize(self, images: list[NumpyImage], general: GeneralData) -> None:
        pass

    def reset(self) -> None:
        pass
