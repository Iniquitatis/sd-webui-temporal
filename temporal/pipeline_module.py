from typing import Any, Optional, Type
from uuid import uuid4

from temporal.general_data import GeneralData
from temporal.meta.configurable import Configurable
from temporal.meta.serializable import SerializableField as Field
from temporal.utils.image import NumpyImage


PIPELINE_MODULES: list[Type["PipelineModule"]] = []


class PipelineModule(Configurable, abstract = True):
    store = PIPELINE_MODULES

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
        }

    def forward(self, image: NumpyImage, general: GeneralData, frame_index: int, seed: int) -> Optional[NumpyImage]:
        return image

    def finalize(self, image: NumpyImage, general: GeneralData) -> None:
        pass

    def reset(self) -> None:
        pass
