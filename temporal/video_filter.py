from abc import abstractmethod
from typing import Iterator

from temporal.object import Field, Object, Static


class VideoFilter(Object, abstract = True):
    name: str = Static("UNDEFINED")

    enabled: bool = Field(True)

    def print(self, fps: int) -> str:
        return ",".join(self.generate(fps))

    @abstractmethod
    def generate(self, fps: int) -> Iterator[str]:
        raise NotImplementedError


def make_filter(inputs: list[str], outputs: list[str], name: str, *args: int | float | str, **kwargs: int | float | str) -> str:
    def sanitize(value: int | float | str) -> str:
        if isinstance(value, (int, float)):
            return str(value)
        else:
            return (
                value
                .replace("\\", "\\\\")
                .replace(":", "\\:")
                .replace("'", "\\'")
            )

    def generate_io(lst: list[str]) -> Iterator[str]:
        for slot in lst:
            yield f"[{slot}]"

    def generate_args() -> Iterator[str]:
        for value in args:
            yield sanitize(value)

        for key, value in kwargs.items():
            yield f"{key}={sanitize(value)}"

    return f"{''.join(generate_io(inputs))}{name}='{':'.join(generate_args())}'{''.join(generate_io(outputs))}"
