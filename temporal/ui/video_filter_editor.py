from typing import Iterator

from temporal.ui import ReadData, UIThing, UpdateData, UpdateRequest, Widget
from temporal.ui.configurable_param_editor import ConfigurableParamEditor
from temporal.ui.reorderable_list import ReorderableAccordion
from temporal.video_filters import VideoFilter


class VideoFilterEditor(Widget):
    def __init__(
        self,
        value: VideoFilter = VideoFilter(),
    ) -> None:
        super().__init__()

        self.type = value.__class__

        with ReorderableAccordion(label = value.name, value = value.enabled, open = False) as self._enabled:
            self._params = {
                key: ConfigurableParamEditor(param = param, value = getattr(value, key))
                for key, param in value.__params__.items()
            }

    @property
    def dependencies(self) -> Iterator[UIThing]:
        yield self._enabled
        yield from self._params.values()

    def read(self, data: ReadData) -> VideoFilter:
        return self.type(
            enabled = data[self._enabled],
            **{key: data[widget] for key, widget in self._params.items()},
        )

    def update(self, data: UpdateData) -> UpdateRequest:
        result: UpdateRequest = {}

        if isinstance(value := data.get("value", None), VideoFilter):
            result[self._enabled] = {"value": value.enabled}
            result |= {widget: {"value": getattr(value, key)} for key, widget in self._params.items()}

        return result
