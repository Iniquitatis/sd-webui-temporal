from typing import Iterator

import gradio as gr

from temporal.noise import Noise
from temporal.ui import Callback, CallbackInputs, CallbackOutputs, ReadData, UIThing, UpdateData, UpdateRequest, Widget
from temporal.ui.gradio_widget import GradioWidget
from temporal.ui.seed_editor import SeedEditor
from temporal.utils.image import NumpyImage
from temporal.utils.numpy import generate_value_noise


class NoiseEditor(Widget):
    def __init__(
        self,
        label: str = "",
        value: Noise = Noise(),
        visible: bool = True,
    ) -> None:
        super().__init__()

        with GradioWidget(gr.Row, visible = visible, elem_classes = ["temporal-gap"]) as self._row:
            self._preview = GradioWidget(gr.Image, label = self._format_label(label, "Preview"), type = "numpy", image_mode = "RGB", value = self._generate_preview_texture(value))

            with GradioWidget(gr.Column):
                self._scale = GradioWidget(gr.Slider, label = self._format_label(label, "Scale"), minimum = 1, maximum = 1024, step = 1, value = value.scale)
                self._octaves = GradioWidget(gr.Slider, label = self._format_label(label, "Octaves"), minimum = 1, maximum = 10, step = 1, value = value.octaves)
                self._lacunarity = GradioWidget(gr.Slider, label = self._format_label(label, "Lacunarity"), minimum = 0.01, maximum = 4.0, step = 0.01, value = value.lacunarity)
                self._persistence = GradioWidget(gr.Slider, label = self._format_label(label, "Persistence"), minimum = 0.0, maximum = 1.0, step = 0.01, value = value.persistence)
                self._seed = SeedEditor(label = self._format_label(label, "Seed"), value = value.seed)
                self._use_global_seed = GradioWidget(gr.Checkbox, label = self._format_label(label, "Use global seed"), value = value.use_global_seed)

        @self.callback("change", [self], [self._preview])
        def _(inputs: CallbackInputs) -> CallbackOutputs:
            return {self._preview: {"value": self._generate_preview_texture(inputs[self])}}

    @property
    def dependencies(self) -> Iterator[UIThing]:
        yield self._scale
        yield self._octaves
        yield self._lacunarity
        yield self._persistence
        yield self._seed
        yield self._use_global_seed

    def read(self, data: ReadData) -> Noise:
        return Noise(
            data[self._scale],
            data[self._octaves],
            data[self._lacunarity],
            data[self._persistence],
            data[self._seed],
            data[self._use_global_seed],
        )

    def update(self, data: UpdateData) -> UpdateRequest:
        result: UpdateRequest = {}

        if isinstance(visible := data.get("visible", None), bool):
            result[self._row] = {"visible": visible}

        if isinstance(value := data.get("value", None), Noise):
            result[self._scale] = {"value": value.scale}
            result[self._octaves] = {"value": value.octaves}
            result[self._lacunarity] = {"value": value.lacunarity}
            result[self._persistence] = {"value": value.persistence}
            result[self._seed] = {"value": value.seed}
            result[self._use_global_seed] = {"value": value.use_global_seed}

        return result

    def setup_callback(self, callback: Callback) -> None:
        self._scale.setup_callback(callback)
        self._octaves.setup_callback(callback)
        self._lacunarity.setup_callback(callback)
        self._persistence.setup_callback(callback)
        self._seed.setup_callback(callback)
        self._use_global_seed.setup_callback(callback)

    def _generate_preview_texture(self, noise: Noise) -> NumpyImage:
        return generate_value_noise(
            (256, 256, 3),
            noise.scale,
            noise.octaves,
            noise.lacunarity,
            noise.persistence,
            noise.seed,
        )
