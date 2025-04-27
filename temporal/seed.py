from random import randint
from typing_extensions import Self

from temporal.object import Field, Object
from temporal.serialization import JSONValue, SerializationParams


class Seed(Object):
    value: int = Field(-1)

    @property
    def fixed_value(self) -> int:
        if self.value == -1:
            self.value = randint(0, 0x7fffffff)

        return self.value

    @classmethod
    def from_json(cls, data: JSONValue, params: SerializationParams = SerializationParams()) -> Self:
        if not isinstance(data, int):
            raise ValueError

        return cls(data)

    def to_json(self, params: SerializationParams = SerializationParams()) -> JSONValue:
        return self.value
