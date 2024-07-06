from typing import Iterator

from temporal.blend_modes import BLEND_MODES, BlendMode, NormalBlendMode
from temporal.ui import Callback, ReadData, UIThing, UpdateData, UpdateRequest, Widget
from temporal.ui.dropdown import Dropdown


class BlendModeSelector(Widget):
    def __init__(
        self,
        label: str = "",
        value: BlendMode = NormalBlendMode(),
    ) -> None:
        super().__init__()

        self._selector = Dropdown(label = label, choices = [(x.id, x.name) for x in BLEND_MODES.values()], value = value.id)

    @property
    def dependencies(self) -> Iterator[UIThing]:
        yield self._selector

    def read(self, data: ReadData) -> BlendMode:
        return BLEND_MODES[data[self._selector]]()

    def update(self, data: UpdateData) -> UpdateRequest:
        result: UpdateRequest = {}

        if isinstance(value := data.get("value", None), BlendMode):
            result[self._selector] = {"value": value.id}

        return result

    def setup_callback(self, callback: Callback) -> None:
        self._selector.setup_callback(callback)
