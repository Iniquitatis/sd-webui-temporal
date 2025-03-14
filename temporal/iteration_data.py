from typing import Optional

from temporal.object import Field, Object
from temporal.utils.image import NumpyImage


class IterationData(Object):
    image: Optional[NumpyImage] = Field(None)
    index: int = Field(1)
    step: int = Field(0)
