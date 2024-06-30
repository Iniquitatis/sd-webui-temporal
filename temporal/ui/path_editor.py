from pathlib import Path
from typing import Callable, Iterator

import gradio as gr

from temporal.ui import Callback, ReadData, UIThing, UpdateData, UpdateRequest, Widget
from temporal.ui.gradio_widget import GradioWidget


class PathEditor(Widget):
    def __init__(
        self,
        label: str = "",
        value: Path | Callable[[], Path] = Path(),
    ) -> None:
        super().__init__()

        self._instance = GradioWidget(gr.Textbox, label = self._format_label(label), value = value.as_posix() if isinstance(value, Path) else lambda: value().as_posix())

    @property
    def dependencies(self) -> Iterator[UIThing]:
        yield self._instance

    def read(self, data: ReadData) -> Path:
        return Path(data[self._instance])

    def update(self, data: UpdateData) -> UpdateRequest:
        result: UpdateRequest = {self._instance: data}

        if isinstance(value := data.pop("value", None), Path):
            result[self._instance]["value"] = value.as_posix()

        return result

    def setup_callback(self, callback: Callback) -> None:
        self._instance.setup_callback(callback)
