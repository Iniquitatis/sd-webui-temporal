from typing import Any, Optional, Type

from temporal.general_data import GeneralData
from temporal.meta.configurable import Configurable
from temporal.meta.serializable import SerializableField as Field
from temporal.utils.collection import find_by_predicate
from temporal.utils.image import NumpyImage


PIPELINE_MODULES: list[Type["PipelineModule"]] = []


class PipelineModule(Configurable, abstract = True):
    store = PIPELINE_MODULES

    icon: str = "\U00002699\ufe0f"

    enabled: bool = Field(False)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "PipelineModule":
        id = data.pop("id", "")

        if type := find_by_predicate(PIPELINE_MODULES, lambda x: x.id == id):
            return type.from_json(data)
        else:
            return super().from_json(data)

    @classmethod
    def schema(cls) -> dict[str, Any]:
        from temporal.pipeline_modules.filtering import ImageFilter

        return super().schema() | {
            "icon": cls.icon,
            "is_filter": issubclass(cls, ImageFilter),
        }

    def to_json(self) -> dict[str, Any]:
        return {"id": self.id} | super().to_json()

    def forward(self, images: list[NumpyImage], general: GeneralData, frame_index: int, seed: int) -> Optional[list[NumpyImage]]:
        return images

    def finalize(self, images: list[NumpyImage], general: GeneralData) -> None:
        pass

    def reset(self) -> None:
        pass
