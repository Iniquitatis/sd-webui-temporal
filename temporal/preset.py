from typing import Any

from temporal.meta.serializable import Serializable, SerializableField as Field


class Preset(Serializable):
    data: dict[str, Any] = Field(factory = dict)
