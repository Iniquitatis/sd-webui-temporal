from typing import Optional

from temporal.meta.serializable import Serializable, SerializableField as Field
from temporal.utils.image import NumpyImage


class IterationData(Serializable):
    image: Optional[NumpyImage] = Field(None, variant = "image")
    index: int = Field(1)
    step: int = Field(0)
