from pathlib import Path
from typing import Optional

from temporal.object import Field, Object
from temporal.seed import Seed
from temporal.shared import shared
from temporal.utils.image import NumpyImage
from temporal.vector import IntVector


class GeneralData(Object):
    name: str = Field("untitled", name = "Name", display = "box")
    description: str = Field("", name = "Description", display = "area")
    mode: str = Field("loop", name = "Mode", choices = {"loop": "Loop", "recursion": "Recursion"}, display = "radio")
    initial_image: Optional[NumpyImage] = Field(None, name = "Initial image")
    seed: Seed = Field(Seed, name = "Seed")

    @property
    def image_size(self) -> IntVector:
        if self.initial_image is not None:
            return IntVector(*reversed(self.initial_image.shape[:2]))
        else:
            return IntVector(0, 0)

    @property
    def path(self) -> Path:
        return shared.settings.fs.project_dir / self.name
