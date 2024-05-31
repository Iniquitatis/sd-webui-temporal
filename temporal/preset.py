from typing import Any

from temporal.meta.serializable import Serializable, field


class Preset(Serializable):
    data: dict[str, Any] = field(factory = dict)
