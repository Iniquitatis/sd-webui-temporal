from typing import Iterator
from typing_extensions import Self

import numpy as np

from modules.object import Field, Object
from modules.utils.numpy import FloatArray, FloatType, IntArray, IntType


class IntVector(Object):
    x: int = Field(0)
    y: int = Field(0)

    def __iter__(self) -> Iterator[int]:
        for key in self.__fields__.keys():
            yield getattr(self, key)

    @classmethod
    def from_numpy(cls, arr: IntArray) -> Self:
        return cls(*arr)

    def to_numpy(self) -> IntArray:
        return np.fromiter(self, IntType)


class FloatVector(Object):
    x: float = Field(0.0)
    y: float = Field(0.0)

    def __iter__(self) -> Iterator[float]:
        for key in self.__fields__.keys():
            yield getattr(self, key)

    @classmethod
    def from_numpy(cls, arr: FloatArray) -> Self:
        return cls(*arr)

    def to_numpy(self) -> FloatArray:
        return np.fromiter(self, FloatType)
