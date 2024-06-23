from typing import Iterator

import gradio as gr

from temporal.image_source import ImageSource
from temporal.ui import ReadData, ResolvedCallback, ResolvedCallbackInputs, ResolvedCallbackOutputs, UIThing, UpdateData, UpdateRequest, Widget
from temporal.ui.gradio_widget import GradioWidget
from temporal.ui.video_box import VideoBox
from temporal.utils.image import ensure_image_dims


class ImageSourceEditor(Widget):
    def __init__(
        self,
        label: str = "",
        channels: int = 3,
        value: ImageSource = ImageSource(),
    ) -> None:
        super().__init__()

        self._type = GradioWidget(gr.Radio, label = self._format_label(label, "Type"), choices = ["image", "initial_image", "video"], value = value.type)
        self._image = GradioWidget(gr.Image, label = self._format_label(label, "Image"), type = "numpy", image_mode = "RGBA" if channels == 4 else "RGB", value = value.image, visible = value.type == "image")
        self._video = VideoBox(label = self._format_label(label, "Video"), value = value.video, visible = value.type == "video")

        @self._type.callback("change", [self._type, self._image, self._video], [self._image, self._video])
        def _(inputs: ResolvedCallbackInputs) -> ResolvedCallbackOutputs:
            outputs: ResolvedCallbackOutputs = {
                self._image: {"visible": False},
                self._video: {"visible": False},
            }

            type = inputs[self._type]

            if type == "image":
                outputs[self._image]["visible"] = True
            elif type == "initial_image":
                pass
            elif type == "video":
                outputs[self._video]["visible"] = True
            else:
                raise NotImplementedError

            return outputs

    @property
    def dependencies(self) -> Iterator[UIThing]:
        yield self._type
        yield self._image
        yield self._video

    def read(self, data: ReadData) -> ImageSource:
        type = data[self._type]
        value = None

        if type == "image":
            value = data[self._image]
        elif type == "initial_image":
            pass
        elif type == "video":
            value = data[self._video]
        else:
            raise NotImplementedError

        return ImageSource(type, value)

    def update(self, data: UpdateData) -> UpdateRequest:
        result: UpdateRequest = {}

        if isinstance(value := data.get("value", None), ImageSource):
            result[self._type] = {"value": value.type}
            result[self._image] = {"visible": False, "value": None}
            result[self._video] = {"visible": False, "value": None}

            if value.type == "image":
                result[self._image]["visible"] = True
                result[self._image]["value"] = ensure_image_dims(value.image, self._image._instance.image_mode) if value.image is not None else None
            elif value.type == "initial_image":
                pass
            elif value.type == "video":
                result[self._video]["visible"] = True
                result[self._video]["value"] = value.video
            else:
                raise NotImplementedError

        return result

    def setup_callback(self, callback: ResolvedCallback) -> None:
        self._type.setup_callback(callback)
        self._image.setup_callback(callback)
        self._video.setup_callback(callback)
