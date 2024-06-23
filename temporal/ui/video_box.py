from pathlib import Path
from typing import Iterator, Optional

import gradio as gr

from temporal.ui import ReadData, ResolvedCallback, UIThing, UpdateData, UpdateRequest, Widget
from temporal.ui.gradio_widget import GradioWidget
from temporal.video import Video


class VideoBox(Widget):
    def __init__(
        self,
        label: str = "",
        value: Optional[Video] = None,
        visible: bool = True,
    ) -> None:
        super().__init__()

        self._instance = GradioWidget(gr.Video, label = self._format_label(label), value = value.path if value is not None else None, visible = visible)

    @property
    def dependencies(self) -> Iterator[UIThing]:
        yield self._instance

    def read(self, data: ReadData) -> Optional[Video]:
        if data[self._instance] is not None:
            return Video.load_from_path(Path(data[self._instance]))

    def update(self, data: UpdateData) -> UpdateRequest:
        result: UpdateRequest = {self._instance: data}

        if "value" in data:
            if isinstance(value := data.pop("value", None), Video):
                result[self._instance]["value"] = value.store_on_disk()
            else:
                result[self._instance]["value"] = value

        return result

    def setup_callback(self, callback: ResolvedCallback) -> None:
        self._instance.setup_callback(callback)
