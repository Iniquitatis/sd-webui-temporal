from pathlib import Path
from time import sleep
from types import SimpleNamespace

import gradio as gr

from modules import scripts
from modules.ui_components import ToolButton

from temporal.image_generation import generate_project
from temporal.image_preprocessing import PREPROCESSORS
from temporal.interop import EXTENSION_DIR
from temporal.metrics import Metrics
from temporal.presets import delete_preset, presets, refresh_presets, save_preset
from temporal.video_rendering import start_video_render, video_render_queue

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
        stored_elem_dict = {}

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
                stored_elem_dict[key] = elem

            return elem

        with gr.Row():
            elem("preset", gr.Dropdown, label = "Preset", choices = list(presets.keys()), allow_custom_value = True, value = next(iter(presets)) if presets else "", stored = False)
            elem("refresh_presets", ToolButton, value = "\U0001f504")
            elem("load_preset", ToolButton, value = "\U0001f4c2")
            elem("save_preset", ToolButton, value = "\U0001f4be")
            elem("delete_preset", ToolButton, value = "\U0001f5d1\ufe0f")

        with gr.Tab("General"):
            with gr.Accordion("Output"):
                with gr.Row():
                    elem("output_dir", gr.Textbox, label = "Output directory", value = "outputs/temporal")
                    elem("project_subdir", gr.Textbox, label = "Project subdirectory", value = "untitled")

                with gr.Row():
                    elem("frame_count", gr.Number, label = "Frame count", precision = 0, minimum = 1, step = 1, value = 100)
                    elem("save_every_nth_frame", gr.Number, label = "Save every N-th frame", precision = 0, minimum = 1, step = 1, value = 1)

                elem("archive_mode", gr.Checkbox, label = "Archive mode", value = False)

            with gr.Accordion("Rendering"):
                with gr.Row():
                    elem("image_samples", gr.Number, label = "Image samples", precision = 0, minimum = 1, value = 1)
                    elem("batch_size", gr.Number, label = "Batch size", precision = 0, minimum = 1, value = 1)

            with gr.Accordion("Project"):
                elem("start_from_scratch", gr.Checkbox, label = "Start from scratch", value = False)
                elem("load_session", gr.Checkbox, label = "Load session", value = True)
                elem("save_session", gr.Checkbox, label = "Save session", value = True)

        with gr.Tab("Frame Preprocessing"):
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

            with gr.Accordion("Deflickering", open = False):
                elem("video_deflickering_enabled", gr.Checkbox, label = "Enabled", value = False)
                elem("video_deflickering_frames", gr.Slider, label = "Frames", minimum = 2, maximum = 120, step = 1, value = 60)

            with gr.Accordion("Interpolation", open = False):
                elem("video_interpolation_enabled", gr.Checkbox, label = "Enabled", value = False)
                elem("video_interpolation_fps", gr.Slider, label = "Frames per second", minimum = 1, maximum = 60, step = 1, value = 60)
                elem("video_interpolation_mb_subframes", gr.Slider, label = "Motion blur subframes", minimum = 0, maximum = 15, step = 1, value = 0)

            with gr.Accordion("Temporal blurring", open = False):
                elem("video_temporal_blurring_enabled", gr.Checkbox, label = "Enabled", value = False)
                elem("video_temporal_blurring_radius", gr.Slider, label = "Radius", minimum = 1, maximum = 10, step = 1, value = 1)
                elem("video_temporal_blurring_easing", gr.Slider, label = "Easing", minimum = 0.0, maximum = 16.0, step = 0.1, value = 0.0)

            with gr.Accordion("Scaling", open = False):
                elem("video_scaling_enabled", gr.Checkbox, label = "Enabled", value = False)

                with gr.Row():
                    elem("video_scaling_width", gr.Slider, label = "Width", minimum = 16, maximum = 2560, step = 16, value = 512)
                    elem("video_scaling_height", gr.Slider, label = "Height", minimum = 16, maximum = 2560, step = 16, value = 512)

                elem("video_scaling_padded", gr.Checkbox, label = "Padded", value = False)

            with gr.Accordion("Frame number overlay", open = False):
                elem("video_frame_num_overlay_enabled", gr.Checkbox, label = "Enabled", value = False)
                elem("video_frame_num_overlay_font_size", gr.Number, label = "Font size", precision = 0, minimum = 1, maximum = 144, step = 1, value = 16)

                with gr.Row():
                    elem("video_frame_num_overlay_text_color", gr.ColorPicker, label = "Text color", value = "#ffffff")
                    elem("video_frame_num_overlay_text_alpha", gr.Slider, label = "Text alpha", minimum = 0.0, maximum = 1.0, step = 0.01, value = 1.0)

                with gr.Row():
                    elem("video_frame_num_overlay_shadow_color", gr.ColorPicker, label = "Shadow color", value = "#000000")
                    elem("video_frame_num_overlay_shadow_alpha", gr.Slider, label = "Shadow alpha", minimum = 0.0, maximum = 1.0, step = 0.01, value = 1.0)

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
                ("tab_general.md", "General tab"),
                ("tab_frame_preprocessing.md", "Frame Preprocessing tab"),
                ("tab_video_rendering.md", "Video Rendering tab"),
                ("tab_metrics.md", "Metrics tab"),
                ("additional_notes.md", "Additional notes"),
            ]:
                with open(EXTENSION_DIR / f"docs/temporal/{file_name}", "r", encoding = "utf-8") as file:
                    text = file.read()

                with gr.Accordion(title, open = False):
                    gr.Markdown(text)

        def refresh_presets_callback():
            refresh_presets()
            return gr.update(choices = list(presets.keys()))

        def load_preset_callback(preset, *args):
            ext_params = self._unpack_ext_params(*args)
            return [gr.update(value = presets.get(preset, {}).get(k, v)) for k, v in vars(ext_params).items()]

        def save_preset_callback(preset, *args):
            ext_params = self._unpack_ext_params(*args)
            save_preset(preset, ext_params)
            return gr.update(choices = list(presets.keys()), value = preset)

        def delete_preset_callback(preset):
            delete_preset(preset)
            return gr.update(choices = list(presets.keys()), value = next(iter(presets)) if presets else "")

        def make_render_callback(is_final):
            def callback(*args):
                yield gr.update(interactive = False), gr.update(interactive = False), None

                ext_params = self._unpack_ext_params(*args)

                start_video_render(ext_params, is_final)

                while video_render_queue.busy:
                    sleep(1)

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
        elems.render_draft.click(make_render_callback(False), inputs = list(stored_elem_dict.values()), outputs = [elems.render_draft, elems.render_final, elems.video_preview])
        elems.render_final.click(make_render_callback(True), inputs = list(stored_elem_dict.values()), outputs = [elems.render_draft, elems.render_final, elems.video_preview])
        elems.render_plots.click(render_plots_callback, inputs = list(stored_elem_dict.values()), outputs = [elems.metrics_plots])

        self._elem_names = list(stored_elem_dict.keys())

        return list(stored_elem_dict.values())

    def run(self, p, *args):
        ext_params = self._unpack_ext_params(*args)
        processed = generate_project(p, ext_params)

        if ext_params.render_draft_on_finish:
            start_video_render(ext_params, False)

        if ext_params.render_final_on_finish:
            start_video_render(ext_params, True)

        return processed

    def _unpack_ext_params(self, *args):
        return SimpleNamespace(**{name: arg for name, arg in zip(self._elem_names, args)})
