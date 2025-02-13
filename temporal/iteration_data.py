from temporal.meta.serializable import Serializable, SerializableField as Field
from temporal.utils.image import NumpyImage


class IterationData(Serializable):
    images: list[NumpyImage] = Field(factory = list)
    index: int = Field(1)
    step: int = Field(0)
