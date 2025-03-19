from pathlib import Path
from random import randint
from typing import Any, Optional

from temporal.object import Field, Object
from temporal.shared import shared
from temporal.utils.image import NumpyImage
from temporal.vector import IntVector


class GeneralData(Object):
    # FIXME: Path should be constructed dynamically by getting the global
    # project directory and the name (sanitized, of course)
    # TODO: Set to a temporary directory until it's saved. On save, it should be
    # moved into an appropriate directory (shared.settings.fs.project_dir).
    path: Path = Field(lambda: shared.settings.fs.project_dir / "untitled", flags = {"runtime"})
    name: str = Field("untitled")
    description: str = Field("")
    initial_image: Optional[NumpyImage] = Field(None)
    seed: int = Field(-1)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        if self.seed == -1:
            self.seed = randint(0, 0x7fffffff)

    @property
    def image_size(self) -> IntVector:
        if self.initial_image is not None:
            return IntVector(*reversed(self.initial_image.shape[:2]))
        else:
            return IntVector(0, 0)
