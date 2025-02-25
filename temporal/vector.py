from typing import Iterator

import numpy as np

from temporal.meta.serializable import Serializable, SerializableField as Field
from temporal.utils.numpy import FloatArray, FloatType, IntArray, IntType


class IntVector(Serializable):
    x: int = Field(0)
    y: int = Field(0)

    def __iter__(self) -> Iterator[int]:
        yield from self.__dict__.values()

    @classmethod
    def from_numpy(cls, arr: IntArray) -> "IntVector":
        return cls(*arr)

    def to_numpy(self) -> IntArray:
        return np.fromiter(self, IntType)


class FloatVector(Serializable):
    x: float = Field(0.0)
    y: float = Field(0.0)

    def __iter__(self) -> Iterator[float]:
        yield from self.__dict__.values()

    @classmethod
    def from_numpy(cls, arr: FloatArray) -> "FloatVector":
        return cls(*arr)

    def to_numpy(self) -> FloatArray:
        return np.fromiter(self, FloatType)
