from typing import Iterator

import numpy as np

from modules.object import Field, Object
from modules.utils.numpy import FloatArray, FloatType


class Color(Object):
    r: float = Field(0.0)
    g: float = Field(0.0)
    b: float = Field(0.0)
    a: float = Field(1.0)

    def __iter__(self) -> Iterator[float]:
        for key in self.__fields__.keys():
            yield getattr(self, key)

    @classmethod
    def from_hex(cls, hex: str) -> "Color":
        return cls(*(
            int(hex[i:i + 2], 16) / 255.0
            for i in range(1, len(hex), 2)
        ))

    @classmethod
    def from_numpy(cls, arr: FloatArray) -> "Color":
        return cls(*arr)

    def to_hex(self, channels: int = 4) -> str:
        return "#" + "".join([
            f"{round(x * 255.0):02x}"
            for x in self
        ][:channels])

    def to_numpy(self, channels: int = 4) -> FloatArray:
        return np.fromiter(self, FloatType)[:channels]
