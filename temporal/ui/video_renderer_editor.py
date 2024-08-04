from typing import Iterator

import gradio as gr

from temporal.ui import ReadData, UIThing, UpdateData, UpdateRequest, Widget
from temporal.ui.gradio_widget import GradioWidget
from temporal.ui.reorderable_list import ReorderableList
from temporal.ui.video_filter_editor import VideoFilterEditor
from temporal.utils.collection import find_by_predicate, reorder_dict
from temporal.video_renderer import VideoRenderer


class VideoRendererEditor(Widget):
    def __init__(
        self,
        value: VideoRenderer = VideoRenderer(),
    ) -> None:
        super().__init__()

        self._fps = GradioWidget(gr.Slider, label = "Frames per second", minimum = 1, maximum = 60, step = 1, value = value.fps)

        with GradioWidget(gr.Row):
            self._first_frame = GradioWidget(gr.Number, label = "First frame", precision = 0, minimum = 1, step = 1, value = value.first_frame)
            self._last_frame = GradioWidget(gr.Number, label = "Last frame", precision = 0, minimum = 0, step = 1, value = value.last_frame)

        self._frame_stride = GradioWidget(gr.Number, label = "Frame stride", precision = 0, minimum = 1, step = 1, value = value.frame_stride)
        self._looping = GradioWidget(gr.Checkbox, label = "Looping", value = value.looping)

        with ReorderableList() as self._filter_order:
            self._filters = {
                filter.id: VideoFilterEditor(value = filter)
                for filter in value.filters
            }

    @property
    def dependencies(self) -> Iterator[UIThing]:
        yield self._fps
        yield self._first_frame
        yield self._last_frame
        yield self._frame_stride
        yield self._looping
        yield self._filter_order
        yield from self._filters.values()

    def read(self, data: ReadData) -> VideoRenderer:
        return VideoRenderer(
            fps = data[self._fps],
            first_frame = data[self._first_frame],
            last_frame = data[self._last_frame],
            frame_stride = data[self._frame_stride],
            looping = data[self._looping],
            filters = [
                data[widget]
                for widget in reorder_dict(self._filters, [
                    self._filter_ids[x]
                    for x in data[self._filter_order]
                ]).values()
            ],
        )

    def update(self, data: UpdateData) -> UpdateRequest:
        result: UpdateRequest = {}

        if isinstance(value := data.get("value", None), VideoRenderer):
            result[self._fps] = {"value": value.fps}
            result[self._first_frame] = {"value": value.first_frame}
            result[self._last_frame] = {"value": value.last_frame}
            result[self._frame_stride] = {"value": value.frame_stride}
            result[self._looping] = {"value": value.looping}
            result[self._filter_order] = {"value": [self._filter_ids.index(x.id) for x in value.filters]}
            result |= {
                widget: {"value": find_by_predicate(value.filters, lambda x: x.id == id)}
                for id, widget in self._filters.items()
            }

        return result

    @property
    def _filter_ids(self) -> list[str]:
        return list(self._filters.keys())
