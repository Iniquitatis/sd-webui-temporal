from typing import Any

from temporal.object import Field, Object


class Preset(Object):
    data: dict[str, Any] = Field(dict)
