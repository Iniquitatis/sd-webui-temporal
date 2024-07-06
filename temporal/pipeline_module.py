from typing import Optional, Type

from temporal.meta.configurable import Configurable
from temporal.meta.serializable import SerializableField as Field
from temporal.project import Project
from temporal.utils.image import NumpyImage


PIPELINE_MODULES: list[Type["PipelineModule"]] = []


class PipelineModule(Configurable, abstract = True):
    store = PIPELINE_MODULES

    icon: str = "\U00002699"

    enabled: bool = Field(False)

    def forward(self, images: list[NumpyImage], project: Project, frame_index: int, seed: int) -> Optional[list[NumpyImage]]:
        return images

    def finalize(self, images: list[NumpyImage], project: Project) -> None:
        pass

    def reset(self) -> None:
        pass
