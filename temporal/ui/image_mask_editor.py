from typing import Iterator

import gradio as gr

from temporal.image_mask import ImageMask
from temporal.ui import Callback, ReadData, UIThing, UpdateData, UpdateRequest, Widget
from temporal.ui.gradio_widget import GradioWidget


class ImageMaskEditor(Widget):
    def __init__(
        self,
        label: str = "",
        value: ImageMask = ImageMask(),
    ) -> None:
        super().__init__()

        self._image = GradioWidget(gr.Image, label = self._format_label(label, "Image"), type = "numpy", image_mode = "L", interactive = True, value = value.image)
        self._normalized = GradioWidget(gr.Checkbox, label = self._format_label(label, "Normalized"), value = value.normalized)
        self._inverted = GradioWidget(gr.Checkbox, label = self._format_label(label, "Inverted"), value = value.inverted)
        self._blurring = GradioWidget(gr.Slider, label = self._format_label(label, "Blurring"), minimum = 0.0, maximum = 50.0, step = 0.1, value = value.blurring)

    @property
    def dependencies(self) -> Iterator[UIThing]:
        yield self._image
        yield self._normalized
        yield self._inverted
        yield self._blurring

    def read(self, data: ReadData) -> ImageMask:
        return ImageMask(
            data[self._image],
            data[self._normalized],
            data[self._inverted],
            data[self._blurring],
        )

    def update(self, data: UpdateData) -> UpdateRequest:
        result: UpdateRequest = {}

        if isinstance(value := data.pop("value", None), ImageMask):
            result[self._image] = {"value": value.image}
            result[self._normalized] = {"value": value.normalized}
            result[self._inverted] = {"value": value.inverted}
            result[self._blurring] = {"value": value.blurring}

        return result

    def setup_callback(self, callback: Callback) -> None:
        self._image.setup_callback(callback)
        self._normalized.setup_callback(callback)
        self._inverted.setup_callback(callback)
        self._blurring.setup_callback(callback)
