from typing import Optional

from temporal.meta.serializable import Serializable, SerializableField as Field
from temporal.utils.image import NumpyImage


class ImageMask(Serializable):
    image: Optional[NumpyImage] = Field(None)
    normalized: bool = Field(False)
    inverted: bool = Field(False)
    blurring: float = Field(0.0)
