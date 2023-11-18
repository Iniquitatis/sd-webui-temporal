from pathlib import Path
from time import sleep
from types import SimpleNamespace

import gradio as gr

from modules import scripts

from temporal.image_generation import generate_project
from temporal.image_preprocessing import PREPROCESSORS
from temporal.interop import EXTENSION_DIR
from temporal.metrics import Metrics
from temporal.video_rendering import start_video_render, video_render_queue

class TemporalScript(scripts.Script):
    def title(self):
        return "Temporal"

    def show(self, is_img2img):
        return is_img2img

    def ui(self, is_img2img):
        ue = SimpleNamespace()
        ue_dict = vars(ue)

        labels = set()

        def unique_label(string):
            if string in labels:
                string = unique_label(string + " ")

            labels.add(string)

            return string

        def elem(key, gr_type, *args, **kwargs):
            if "label" in kwargs:
                kwargs["label"] = unique_label(kwargs["label"])

            elem = gr_type(*args, elem_id = self.elem_id(key), **kwargs)
            setattr(ue, key, elem)

            return elem

        with gr.Tab("General"):
            elem("output_dir", gr.Textbox, label = "Output directory", value = "outputs/temporal")
            elem("project_subdir", gr.Textbox, label = "Project subdirectory", value = "untitled")
            elem("frame_count", gr.Number, label = "Frame count", precision = 0, minimum = 1, step = 1, value = 100)
            elem("save_every_nth_frame", gr.Number, label = "Save every N-th frame", precision = 0, minimum = 1, step = 1, value = 1)
            elem("archive_mode", gr.Checkbox, label = "Archive mode", value = False)
            elem("start_from_scratch", gr.Checkbox, label = "Start from scratch", value = False)
            elem("load_session", gr.Checkbox, label = "Load session", value = True)
            elem("save_session", gr.Checkbox, label = "Save session", value = True)

        with gr.Tab("Frame Preprocessing"):
            for key, processor in PREPROCESSORS.items():
                with gr.Accordion(processor.name, open = False):
                    elem(f"{key}_enabled", gr.Checkbox, label = "Enabled")

                    with gr.Row():
                        elem(f"{key}_amount", gr.Slider, label = "Amount", minimum = 0.0, maximum = 1.0, step = 0.01, value = 1.0)
                        elem(f"{key}_amount_relative", gr.Checkbox, label = "Relative", value = False)

                    with gr.Tab("Parameters"):
                        for param in processor.params:
                            elem(f"{key}_{param.key}", param.type, label = param.name, **param.kwargs)

                    with gr.Tab("Mask"):
                        elem(f"{key}_mask", gr.Pil, label = "Mask", image_mode = "L", interactive = True)
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

        def make_render_callback(is_final):
            def callback(*args):
                yield gr.Button.update(interactive = False), gr.Button.update(interactive = False), None

                uv = self._get_ui_values(*args)

                start_video_render(uv, is_final)

                while video_render_queue.busy:
                    sleep(1)

                yield gr.Button.update(interactive = True), gr.Button.update(interactive = True), f"{uv.output_dir}/{uv.project_subdir}-{'final' if is_final else 'draft'}.mp4"

            return callback

        def render_plots_callback(*args):
            uv = self._get_ui_values(*args)
            project_dir = Path(uv.output_dir) / uv.project_subdir
            metrics = Metrics()
            metrics.load(project_dir)
            return gr.Gallery.update(value = list(metrics.plot(project_dir)))

        ue.render_draft.click(make_render_callback(False), inputs = list(ue_dict.values()), outputs = [ue.render_draft, ue.render_final, ue.video_preview])
        ue.render_final.click(make_render_callback(True), inputs = list(ue_dict.values()), outputs = [ue.render_draft, ue.render_final, ue.video_preview])
        ue.render_plots.click(render_plots_callback, inputs = list(ue_dict.values()), outputs = [ue.metrics_plots])

        self._ui_element_names = list(ue_dict.keys())

        return list(ue_dict.values())

    def run(self, p, *args):
        uv = self._get_ui_values(*args)
        processed = generate_project(p, uv)

        if uv.render_draft_on_finish:
            start_video_render(uv, False)

        if uv.render_final_on_finish:
            start_video_render(uv, True)

        return processed

    def _get_ui_values(self, *args):
        return SimpleNamespace(**{name: arg for name, arg in zip(self._ui_element_names, args)})
