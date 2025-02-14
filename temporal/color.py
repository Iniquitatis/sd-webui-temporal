from typing import Iterator

import numpy as np
from numpy.typing import NDArray

from temporal.meta.serializable import Serializable, SerializableField as Field


class Color(Serializable):
    r: float = Field(0.0)
    g: float = Field(0.0)
    b: float = Field(0.0)
    a: float = Field(1.0)

    def __iter__(self) -> Iterator[float]:
        yield from self.__dict__.values()

    @classmethod
    def from_hex(cls, hex: str) -> "Color":
        return cls(*(
            int(hex[i:i + 2], 16) / 255.0
            for i in range(1, len(hex), 2)
        ))

    @classmethod
    def from_numpy(cls, arr: NDArray[np.float64]) -> "Color":
        return cls(*arr)

    def to_hex(self, channels: int = 4) -> str:
        return "#" + "".join([
            f"{round(x * 255.0):02x}"
            for x in self
        ][:channels])

    def to_numpy(self, channels: int = 4) -> NDArray[np.float64]:
        return np.fromiter(self, np.float64)[:channels]
