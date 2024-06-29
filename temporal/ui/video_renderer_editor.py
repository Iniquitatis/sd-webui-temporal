from typing import Iterator

import gradio as gr

from temporal.ui import ReadData, UIThing, UpdateData, UpdateRequest, Widget
from temporal.ui.gradio_widget import GradioWidget
from temporal.ui.module_list import ModuleList
from temporal.ui.video_filter_editor import VideoFilterEditor
from temporal.utils.collection import reorder_dict
from temporal.video_renderer import VideoRenderer


class VideoRendererEditor(Widget):
    def __init__(
        self,
        value: VideoRenderer = VideoRenderer(),
    ) -> None:
        super().__init__()

        self._fps = GradioWidget(gr.Slider, label = "Frames per second", minimum = 1, maximum = 60, step = 1, value = value.fps)
        self._looping = GradioWidget(gr.Checkbox, label = "Looping", value = value.looping)

        with ModuleList(keys = value.filters.keys()) as self._filter_order:
            self._filters = {
                id: VideoFilterEditor(value = filter)
                for id, filter in value.filters.items()
            }

    @property
    def dependencies(self) -> Iterator[UIThing]:
        yield self._fps
        yield self._looping
        yield self._filter_order
        yield from self._filters.values()

    def read(self, data: ReadData) -> VideoRenderer:
        return VideoRenderer(
            fps = data[self._fps],
            looping = data[self._looping],
            filters = {key: data[widget] for key, widget in reorder_dict(self._filters, data[self._filter_order]).items()},
        )

    def update(self, data: UpdateData) -> UpdateRequest:
        result: UpdateRequest = {}

        if isinstance(value := data.get("value", None), VideoRenderer):
            result[self._fps] = {"value": value.fps}
            result[self._looping] = {"value": value.looping}
            result[self._filter_order] = {"value": list(value.filters.keys())}
            result |= {widget: {"value": value.filters[key]} for key, widget in self._filters.items()}

        return result
