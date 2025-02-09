from typing import Any, Optional, Type

from temporal.meta.configurable import Configurable
from temporal.meta.serializable import SerializableField as Field
from temporal.project import Project
from temporal.utils.collection import find_by_predicate
from temporal.utils.image import NumpyImage


PIPELINE_MODULES: list[Type["PipelineModule"]] = []


class PipelineModule(Configurable, abstract = True):
    store = PIPELINE_MODULES

    icon: str = "\U00002699"

    enabled: bool = Field(False)

    @staticmethod
    def from_json(data: dict[str, Any]) -> "PipelineModule":
        id = data.pop("id")

        if not (type := find_by_predicate(PIPELINE_MODULES, lambda x: x.id == id)):
            raise ValueError

        result = type()
        result.enabled = data.pop("enabled", True)

        for key, param in data.items():
            if key in result.__dict__:
                result.__dict__[key] = param

        return result

    @classmethod
    def schema(cls) -> dict[str, Any]:
        from temporal.pipeline_modules.filtering import ImageFilter

        return super().schema() | {
            "icon": cls.icon,
            "is_filter": issubclass(cls, ImageFilter),
        }

    def forward(self, images: list[NumpyImage], project: Project, frame_index: int, seed: int) -> Optional[list[NumpyImage]]:
        return images

    def finalize(self, images: list[NumpyImage], project: Project) -> None:
        pass

    def reset(self) -> None:
        pass
