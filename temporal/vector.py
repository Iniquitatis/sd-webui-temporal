from typing import Iterator

import numpy as np
from numpy.typing import NDArray

from temporal.meta.serializable import Serializable, SerializableField as Field


class IntVector(Serializable):
    x: int = Field(0)
    y: int = Field(0)

    def __iter__(self) -> Iterator[int]:
        yield from self.__dict__.values()

    @classmethod
    def from_numpy(cls, arr: NDArray[np.int32]) -> "IntVector":
        return cls(*arr)

    def to_numpy(self) -> NDArray[np.int32]:
        return np.fromiter(self, np.int32)


class FloatVector(Serializable):
    x: float = Field(0.0)
    y: float = Field(0.0)

    def __iter__(self) -> Iterator[float]:
        yield from self.__dict__.values()

    @classmethod
    def from_numpy(cls, arr: NDArray[np.float64]) -> "FloatVector":
        return cls(*arr)

    def to_numpy(self) -> NDArray[np.float64]:
        return np.fromiter(self, np.float64)
