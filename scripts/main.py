from pathlib import Path
from types import SimpleNamespace

import gradio as gr

from modules import scripts
from modules.ui_components import ToolButton

from temporal.collection_utils import get_first_element
from temporal.fs import load_text
from temporal.image_generation import generate_image, generate_sequence
from temporal.image_preprocessing import PREPROCESSORS
from temporal.interop import EXTENSION_DIR
from temporal.metrics import Metrics
from temporal.presets import delete_preset, load_preset, preset_names, refresh_presets, save_preset
from temporal.time_utils import wait_until
from temporal.video_filtering import FILTERS
from temporal.video_rendering import start_video_render, video_render_queue

MODES = dict(
    image = SimpleNamespace(
        func = generate_image,
        hidden_elems = [
            "project_subdir",
            "save_every_nth_frame",
            "archive_mode",
            "load_parameters",
            "continue_from_last_frame",
            "render_draft_on_finish",
            "render_final_on_finish",
            "render_draft",
            "render_final",
            "metrics_enabled",
            "render_plots",
        ],
    ),
    sequence = SimpleNamespace(
        func = generate_sequence,
        hidden_elems = [],
    ),
)

class TemporalScript(scripts.Script):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        refresh_presets()

    def title(self):
        return "Temporal"

    def show(self, is_img2img):
        return is_img2img

    def ui(self, is_img2img):
        labels = set()

        def unique_label(string):
            if string in labels:
                string = unique_label(string + " ")

            labels.add(string)

            return string

        elems = SimpleNamespace()
        elem_dict = vars(elems)
        stored_elems = SimpleNamespace()
        stored_elem_dict = vars(stored_elems)

        def elem(key, gr_type, *args, stored = True, **kwargs):
            if "label" in kwargs:
                kwargs["label"] = unique_label(kwargs["label"])

            elem = gr_type(*args, elem_id = self.elem_id(key), **kwargs)
            setattr(elems, key, elem)

            if stored and gr_type in [
                gr.Checkbox,
                gr.Code,
                gr.ColorPicker,
                gr.Dropdown,
                gr.Image,
                gr.Number,
                gr.Pil,
                gr.Slider,
                gr.Textbox,
            ]:
                setattr(stored_elems, key, elem)

            return elem

        with gr.Row():
            elem("preset", gr.Dropdown, label = "Preset", choices = preset_names, allow_custom_value = True, value = get_first_element(preset_names, ""), stored = False)
            elem("refresh_presets", ToolButton, value = "\U0001f504")
            elem("load_preset", ToolButton, value = "\U0001f4c2")
            elem("save_preset", ToolButton, value = "\U0001f4be")
            elem("delete_preset", ToolButton, value = "\U0001f5d1\ufe0f")

        elem("mode", gr.Dropdown, label = "Mode", choices = list(MODES.keys()), value = "sequence")

        with gr.Tab("General"):
            with gr.Accordion("Output"):
                with gr.Row():
                    elem("output_dir", gr.Textbox, label = "Output directory", value = "outputs/temporal")
                    elem("project_subdir", gr.Textbox, label = "Project subdirectory", value = "untitled")

                with gr.Row():
                    elem("frame_count", gr.Number, label = "Frame count", precision = 0, minimum = 1, step = 1, value = 100)
                    elem("save_every_nth_frame", gr.Number, label = "Save every N-th frame", precision = 0, minimum = 1, step = 1, value = 1)

                elem("archive_mode", gr.Checkbox, label = "Archive mode", value = False)

            with gr.Accordion("Processing"):
                elem("noise_for_first_frame", gr.Checkbox, label = "Noise for first frame", value = False)
                elem("use_sd", gr.Checkbox, label = "Use Stable Diffusion", value = True)

            with gr.Accordion("Multisampling"):
                with gr.Row():
                    elem("multisampling_samples", gr.Number, label = "Sample count", precision = 0, minimum = 1, value = 1)
                    elem("multisampling_batch_size", gr.Number, label = "Batch size", precision = 0, minimum = 1, value = 1)

                elem("multisampling_trimming", gr.Slider, label = "Trimming", minimum = 0.0, maximum = 0.5, step = 0.01, value = 0.0)
                elem("multisampling_easing", gr.Slider, label = "Easing", minimum = 0.0, maximum = 16.0, step = 0.1, value = 0.0)
                elem("multisampling_preference", gr.Slider, label = "Preference", minimum = -2.0, maximum = 2.0, step = 0.1, value = 0.0)

            with gr.Accordion("Frame merging"):
                elem("frame_merging_frames", gr.Number, label = "Frame count", precision = 0, minimum = 1, step = 1, value = 1)
                elem("frame_merging_trimming", gr.Slider, label = "Trimming", minimum = 0.0, maximum = 0.5, step = 0.01, value = 0.0)
                elem("frame_merging_easing", gr.Slider, label = "Easing", minimum = 0.0, maximum = 16.0, step = 0.1, value = 0.0)
                elem("frame_merging_preference", gr.Slider, label = "Preference", minimum = -2.0, maximum = 2.0, step = 0.1, value = 0.0)

            with gr.Accordion("Project"):
                elem("load_parameters", gr.Checkbox, label = "Load parameters", value = True)
                elem("continue_from_last_frame", gr.Checkbox, label = "Continue from last frame", value = True)

        with gr.Tab("Frame Preprocessing"):
            elem("preprocessing_order", gr.Dropdown, label = "Order", multiselect = True, choices = list(PREPROCESSORS.keys()), value = [])

            for key, processor in PREPROCESSORS.items():
                with gr.Accordion(processor.name, open = False):
                    elem(f"{key}_enabled", gr.Checkbox, label = "Enabled", value = False)

                    with gr.Row():
                        elem(f"{key}_amount", gr.Slider, label = "Amount", minimum = 0.0, maximum = 1.0, step = 0.01, value = 1.0)
                        elem(f"{key}_amount_relative", gr.Checkbox, label = "Relative", value = False)

                    with gr.Tab("Parameters"):
                        for param in processor.params:
                            elem(f"{key}_{param.key}", param.type, label = param.name, **param.kwargs)

                    with gr.Tab("Mask"):
                        elem(f"{key}_mask", gr.Pil, label = "Mask", image_mode = "L", interactive = True)
                        elem(f"{key}_mask_normalized", gr.Checkbox, label = "Normalized", value = False)
                        elem(f"{key}_mask_inverted", gr.Checkbox, label = "Inverted", value = False)
                        elem(f"{key}_mask_blurring", gr.Slider, label = "Blurring", minimum = 0.0, maximum = 50.0, step = 0.1, value = 0.0)

        with gr.Tab("Video Rendering"):
            elem("video_fps", gr.Slider, label = "Frames per second", minimum = 1, maximum = 60, step = 1, value = 30)
            elem("video_looping", gr.Checkbox, label = "Looping", value = False)
            elem("video_filtering_order", gr.Dropdown, label = "Order", multiselect = True, choices = list(FILTERS.keys()), value = [])

            for key, filter in FILTERS.items():
                with gr.Accordion(filter.name, open = False):
                    elem(f"video_{key}_enabled", gr.Checkbox, label = "Enabled", value = False)

                    for param in filter.params:
                        elem(f"video_{key}_{param.key}", param.type, label = param.name, **param.kwargs)

            with gr.Row():
                elem("render_draft_on_finish", gr.Checkbox, label = "Render draft when finished", value = False)
                elem("render_final_on_finish", gr.Checkbox, label = "Render final when finished", value = False)

            with gr.Row():
                elem("render_draft", gr.Button, value = "Render draft")
                elem("render_final", gr.Button, value = "Render final")

            elem("video_preview", gr.Video, label = "Preview", format = "mp4", interactive = False)

        with gr.Tab("Metrics"):
            elem("metrics_enabled", gr.Checkbox, label = "Enabled", value = False)
            elem("metrics_save_plots_every_nth_frame", gr.Number, label = "Save plots every N-th frame", precision = 0, minimum = 1, step = 1, value = 10)
            elem("render_plots", gr.Button, value = "Render plots")
            elem("metrics_plots", gr.Gallery, label = "Plots", columns = 4, object_fit = "contain", preview = True)

        with gr.Tab("Help"):
            for file_name, title in [
                ("main.md", "Main"),
                ("tab_general.md", "General tab"),
                ("tab_frame_preprocessing.md", "Frame Preprocessing tab"),
                ("tab_video_rendering.md", "Video Rendering tab"),
                ("tab_metrics.md", "Metrics tab"),
                ("additional_notes.md", "Additional notes"),
            ]:
                with gr.Accordion(title, open = False):
                    gr.Markdown(load_text(EXTENSION_DIR / "docs" / "temporal" / file_name, ""))

        def refresh_presets_callback():
            refresh_presets()
            return gr.update(choices = preset_names)

        def load_preset_callback(preset, *args):
            ext_params = self._unpack_ext_params(*args)
            load_preset(preset, ext_params)
            return [gr.update(value = v) for v in vars(ext_params).values()]

        def save_preset_callback(preset, *args):
            ext_params = self._unpack_ext_params(*args)
            save_preset(preset, ext_params)
            return gr.update(choices = preset_names, value = preset)

        def delete_preset_callback(preset):
            delete_preset(preset)
            return gr.update(choices = preset_names, value = get_first_element(preset_names, ""))

        def mode_callback(mode):
            # TODO: Tabs cannot be hidden; an error is thrown regarding an inability to send a `Tab` as an input component
            return [gr.update(visible = x not in MODES[mode].hidden_elems) for x in self._elem_names]

        def make_render_callback(is_final):
            def callback(*args):
                yield gr.update(interactive = False), gr.update(interactive = False), None

                ext_params = self._unpack_ext_params(*args)

                start_video_render(ext_params, is_final)
                wait_until(lambda: not video_render_queue.busy)

                yield gr.update(interactive = True), gr.update(interactive = True), f"{ext_params.output_dir}/{ext_params.project_subdir}-{'final' if is_final else 'draft'}.mp4"

            return callback

        def render_plots_callback(*args):
            ext_params = self._unpack_ext_params(*args)
            project_dir = Path(ext_params.output_dir) / ext_params.project_subdir
            metrics = Metrics()
            metrics.load(project_dir)
            return gr.update(value = list(metrics.plot(project_dir)))

        elems.refresh_presets.click(refresh_presets_callback, outputs = elems.preset)
        elems.load_preset.click(load_preset_callback, inputs = [elems.preset] + list(stored_elem_dict.values()), outputs = list(stored_elem_dict.values()))
        elems.save_preset.click(save_preset_callback, inputs = [elems.preset] + list(stored_elem_dict.values()), outputs = elems.preset)
        elems.delete_preset.click(delete_preset_callback, inputs = elems.preset, outputs = elems.preset)
        elems.mode.change(mode_callback, inputs = elems.mode, outputs = list(elem_dict.values()))
        elems.render_draft.click(make_render_callback(False), inputs = list(stored_elem_dict.values()), outputs = [elems.render_draft, elems.render_final, elems.video_preview])
        elems.render_final.click(make_render_callback(True), inputs = list(stored_elem_dict.values()), outputs = [elems.render_draft, elems.render_final, elems.video_preview])
        elems.render_plots.click(render_plots_callback, inputs = list(stored_elem_dict.values()), outputs = [elems.metrics_plots])

        self._elem_names = list(elem_dict.keys())
        self._stored_elem_names = list(stored_elem_dict.keys())

        return list(stored_elem_dict.values())

    def run(self, p, *args):
        ext_params = self._unpack_ext_params(*args)
        processed = MODES[ext_params.mode].func(p, ext_params)

        if ext_params.render_draft_on_finish:
            start_video_render(ext_params, False)

        if ext_params.render_final_on_finish:
            start_video_render(ext_params, True)

        return processed

    def _unpack_ext_params(self, *args):
        return SimpleNamespace(**{name: arg for name, arg in zip(self._stored_elem_names, args)})
