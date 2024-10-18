from typing import Iterator, Literal, Optional

import gradio as gr
import numpy as np

from temporal.ui import Callback, ReadData, UIThing, UpdateData, UpdateRequest, Widget
from temporal.ui.gradio_widget import GradioWidget
from temporal.utils.image import NumpyImage, PILImage, ensure_image_dims, np_to_pil, pil_to_np


class ImageBox(Widget):
    def __init__(
        self,
        label: str = "",
        channels: int = 3,
        value: Optional[PILImage | NumpyImage] = None,
        visible: bool = True,
    ) -> None:
        super().__init__()

        self._channels = channels
        self._instance = GradioWidget(gr.Image, label = self._format_label(label), type = "pil", image_mode = _IMAGE_MODES[channels], value = _prepare_image(value, channels), visible = visible)

    @property
    def dependencies(self) -> Iterator[UIThing]:
        yield self._instance

    def read(self, data: ReadData) -> Optional[NumpyImage]:
        if data[self._instance] is not None:
            return pil_to_np(data[self._instance])

    def update(self, data: UpdateData) -> UpdateRequest:
        result: UpdateRequest = {self._instance: data}

        if "value" in data:
            result[self._instance]["value"] = _prepare_image(data.pop("value"), self._channels)

        return result

    def setup_callback(self, callback: Callback) -> None:
        self._instance.setup_callback(callback)


_IMAGE_MODES: dict[int, Literal["L", "RGB", "RGBA"]] = {
    1: "L",
    3: "RGB",
    4: "RGBA",
}


def _prepare_image(image: Optional[PILImage | NumpyImage], channels: int) -> Optional[PILImage]:
    if isinstance(image, PILImage) and image.mode != _IMAGE_MODES[channels]:
        return np_to_pil(ensure_image_dims(pil_to_np(image), channels = channels))

    elif isinstance(image, np.ndarray):
        return np_to_pil(ensure_image_dims(image, channels = channels))

    else:
        return image
