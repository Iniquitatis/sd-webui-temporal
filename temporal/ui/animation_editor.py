from typing import Any, Iterator

import gradio as gr

from temporal.animation import Animation, Track
from temporal.animation.plotting import plot_animation
from temporal.color import Color
from temporal.ui import CallbackInputs, CallbackOutputs, ReadData, UIThing, UpdateData, UpdateRequest, Widget
from temporal.ui.communication_box import CommunicationBox
from temporal.ui.gradio_widget import GradioWidget


class AnimationEditor(Widget):
    def __init__(
        self,
        value: Animation = Animation(),
    ) -> None:
        super().__init__()

        self._properties = {}

        self._communication = CommunicationBox(
            label = "Communication",
            script = f"temporalUpdateAnimationEditor({self.index}, data)",
            value = _write_to_dict(value),
        )
        self._editor = GradioWidget(gr.HTML, value = f"<temporal-anim-editor class=\"{self.index_class} {self._communication.communication_class}\"></temporal-anim-editor>")
        self._render_graphs = GradioWidget(gr.Button, value = "Render graphs")
        self._graphs = GradioWidget(gr.Gallery, label = "Graphs", columns = 4, object_fit = "contain", preview = True)

        @self._render_graphs.callback("click", [self], [self._graphs])
        def _(inputs: CallbackInputs) -> CallbackOutputs:
            return {self._graphs: {"value": plot_animation(inputs[self])}}

    @property
    def dependencies(self) -> Iterator[UIThing]:
        yield self._communication

    def read(self, data: ReadData) -> Animation:
        return _read_from_dict(data[self._communication])

    def update(self, data: UpdateData) -> UpdateRequest:
        result: UpdateRequest = {}

        if (properties := data.get("properties", None) is not None):
            self._properties = properties

        if (value := data.get("value", None)) is not None:
            result[self._communication] = {"value": _write_to_dict(value)}

        return result


def _read_from_dict(data: dict[str, Any]) -> Animation:
    result = Animation()

    property_data = data.get("properties", {})

    for track_data in data.get("tracks", []):
        name = track_data.get("name", None)

        track = Track()
        track.interpolation = track_data.get("interpolation", "linear")
        track.bounds = track_data.get("bounds", "clamp")

        type = property_data.get(name, None)

        for keyframe_data in track_data.get("keyframes"):
            value_data = keyframe_data.get("value", None)

            if isinstance(value_data, bool) and type == "bool":
                value = value_data
            elif isinstance(value_data, (int, float)) and type == "int":
                value = int(value_data)
            elif isinstance(value_data, (int, float)) and type == "float":
                value = float(value_data)
            elif isinstance(value_data, str) and type == "color":
                value = Color.from_hex(value_data)
            else:
                raise ValueError

            track.add_keyframe(keyframe_data.get("frame", -1), value)

        result.tracks[name] = track

    return result


def _write_to_dict(animation: Animation) -> dict[str, Any]:
    result: dict[str, Any] = {
        "properties": {},
        "tracks": [],
    }

    property_data = result["properties"]

    for name, track in animation.tracks.items():
        track_data: dict[str, Any] = {
            "name": name,
            "interpolation": track.interpolation,
            "bounds": track.bounds,
            "keyframes": [],
        }

        for keyframe in track.keyframes:
            value = keyframe.value

            if isinstance(value, bool):
                property_data[name] = "bool"
                value_data = value
            elif isinstance(value, int):
                property_data[name] = "int"
                value_data = value
            elif isinstance(value, float):
                property_data[name] = "float"
                value_data = value
            elif isinstance(value, Color):
                property_data[name] = "color"
                value_data = value.to_hex(3)
            else:
                raise ValueError

            track_data["keyframes"].append({"frame": keyframe.frame, "value": value_data})

        result["tracks"].append(track_data)

    return result
