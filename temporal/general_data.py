from pathlib import Path
from random import randint
from typing import Any, Optional

from temporal.meta.serializable import Serializable, SerializableField as Field
from temporal.noise import Noise
from temporal.utils.image import NumpyImage
from temporal.vector import IntVector


class GeneralData(Serializable):
    # FIXME: Path should be constructed dynamically by getting the global
    # project directory and the name (sanitized, of course)
    path: Path = Field(Path("outputs/temporal/untitled"), flags = {"runtime"})
    name: str = Field("untitled")
    description: str = Field("")
    initial_image: Optional[NumpyImage] = Field(None, variant = "image")
    initial_noise: Noise = Field(factory = Noise)
    image_size: IntVector = Field(factory = lambda: IntVector(512, 512))
    seed: int = Field(-1)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        if self.seed == -1:
            self.seed = randint(0, 0x7fffffff)
