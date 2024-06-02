from typing import Optional

from temporal.meta.serializable import Serializable, field
from temporal.utils.image import PILImage


class ImageMask(Serializable):
    image: Optional[PILImage] = field(None)
    normalized: bool = field(False)
    inverted: bool = field(False)
    blurring: float = field(0.0)
