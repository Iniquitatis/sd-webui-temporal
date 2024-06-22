import numpy as np
from numpy.typing import NDArray

from temporal.meta.serializable import Serializable, SerializableField as Field


class Color(Serializable):
    r: float = Field(1.0)
    g: float = Field(1.0)
    b: float = Field(1.0)
    a: float = Field(1.0)

    @classmethod
    def from_hex(cls, hex: str) -> "Color":
        parts = [hex[i:i + 2] for i in range(1, 9, 2)]

        return cls(
            int(parts[0], 16) / 255.0,
            int(parts[1], 16) / 255.0,
            int(parts[2], 16) / 255.0,
            int(parts[3], 16) / 255.0 if parts[3] else 1.0,
        )

    @classmethod
    def from_numpy(cls, arr: NDArray[np.float_]) -> "Color":
        return cls(
            arr[0],
            arr[1],
            arr[2],
            arr[3] if arr.shape[0] == 4 else 1.0,
        )

    def to_hex(self, channels: int = 4) -> str:
        return "#" + "".join((
            f"{round(self.r * 255.0):02x}",
            f"{round(self.g * 255.0):02x}",
            f"{round(self.b * 255.0):02x}",
            f"{round(self.a * 255.0):02x}",
        )[:channels])

    def to_numpy(self, channels: int = 4) -> NDArray[np.float_]:
        return np.array((
            self.r,
            self.g,
            self.b,
            self.a,
        )[:channels])
