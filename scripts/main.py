from pathlib import Path
from types import SimpleNamespace

import gradio as gr

from modules import scripts
from modules.sd_samplers import visible_sampler_names
from modules.ui_components import InputAccordion, ToolButton

from temporal.collection_utils import get_first_element
from temporal.fs import load_text
from temporal.image_generation import GENERATION_MODES
from temporal.image_preprocessing import PREPROCESSORS
from temporal.interop import EXTENSION_DIR
from temporal.metrics import Metrics
from temporal.presets import delete_preset, load_preset, preset_names, refresh_presets, save_preset
from temporal.session import saved_ext_param_ids
from temporal.string_utils import match_mask
from temporal.time_utils import wait_until
from temporal.video_filtering import FILTERS
from temporal.video_rendering import start_video_render, video_render_queue

class UI:
    def __init__(self, id_formatter):
        self._id_formatter = id_formatter
        self._elems = {}
        self._ids = []
        self._groups = {}
        self._callbacks = {}
        self._existing_labels = set()

    def parse_ids(self, ids):
        result = []

        for id in ids:
            if id.startswith("group:"):
                _, group = id.split(":")
                result.extend(x for x in self._ids if self.is_in_group(x, group))
            else:
                result.extend(x for x in self._ids if match_mask(x, id))

        return result

    def is_in_group(self, id, group):
        return any(match_mask(x, group) for x in self._groups[id])

    def elem(self, id, constructor, *args, groups = [], **kwargs):
        def unique_label(string):
            if string in self._existing_labels:
                string = unique_label(string + " ")

            self._existing_labels.add(string)

            return string

        if "label" in kwargs:
            kwargs["label"] = unique_label(kwargs["label"])

        elem = constructor(*args, elem_id = self._id_formatter(id), **kwargs)

        if id:
            self._ids.append(id)
            self._elems[id] = elem
            self._groups[id] = ["all"] + groups
            self._callbacks[id] = []

        return elem

    def callback(self, id, event, func, inputs, outputs):
        self._callbacks[id].append((event, func, inputs, outputs))

    def finalize(self, ids):
        for id, callbacks in self._callbacks.items():
            for event, func, inputs, outputs in callbacks:
                event_func = getattr(self._elems[id], event)
                event_func(
                    func,
                    inputs = [self._elems[x] for x in self.parse_ids(inputs)],
                    outputs = [self._elems[x] for x in self.parse_ids(outputs)],
                )

        result = [self._elems[x] for x in self.parse_ids(ids)]

        self._id_formatter = None
        self._elems.clear()
        self._existing_labels.clear()

        return result

    def unpack_values(self, ids, *args):
        return SimpleNamespace(**{name: arg for name, arg in zip(self.parse_ids(ids), args)})

class TemporalScript(scripts.Script):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        refresh_presets()

    def title(self):
        return "Temporal"

    def show(self, is_img2img):
        return is_img2img

    def ui(self, is_img2img):
        self._ui = ui = UI(self.elem_id)

        with ui.elem("", gr.Row):
            def refresh_presets_callback():
                refresh_presets()
                return gr.update(choices = preset_names)

            def load_preset_callback(preset, *args):
                ext_params = ui.unpack_values(["group:params"], *args)
                load_preset(preset, ext_params)
                return [gr.update(value = v) for v in vars(ext_params).values()]

            def save_preset_callback(preset, *args):
                ext_params = ui.unpack_values(["group:params"], *args)
                save_preset(preset, ext_params)
                return gr.update(choices = preset_names, value = preset)

            def delete_preset_callback(preset):
                delete_preset(preset)
                return gr.update(choices = preset_names, value = get_first_element(preset_names, ""))

            ui.elem("preset", gr.Dropdown, label = "Preset", choices = preset_names, allow_custom_value = True, value = get_first_element(preset_names, ""))
            ui.elem("refresh_presets", ToolButton, value = "\U0001f504")
            ui.callback("refresh_presets", "click", refresh_presets_callback, [], ["preset"])
            ui.elem("load_preset", ToolButton, value = "\U0001f4c2")
            ui.callback("load_preset", "click", load_preset_callback, ["preset", "group:params"], ["group:params"])
            ui.elem("save_preset", ToolButton, value = "\U0001f4be")
            ui.callback("save_preset", "click", save_preset_callback, ["preset", "group:params"], ["preset"])
            ui.elem("delete_preset", ToolButton, value = "\U0001f5d1\ufe0f")
            ui.callback("delete_preset", "click", delete_preset_callback, ["preset"], ["preset"])

        def mode_callback(mode):
            return [gr.update(visible = ui.is_in_group(x, f"mode_{mode}")) for x in ui.parse_ids(["group:mode_*"])]

        ui.elem("mode", gr.Dropdown, label = "Mode", choices = list(GENERATION_MODES.keys()), value = "sequence", groups = ["params"])
        ui.callback("mode", "change", mode_callback, ["mode"], ["group:mode_*"])

        with ui.elem("", gr.Tab, label = "General"):
            with ui.elem("", gr.Accordion, label = "Output"):
                with ui.elem("", gr.Row):
                    ui.elem("output_dir", gr.Textbox, label = "Output directory", value = "outputs/temporal", groups = ["params"])
                    ui.elem("project_subdir", gr.Textbox, label = "Project subdirectory", value = "untitled", groups = ["params", "mode_sequence"])

                with ui.elem("", gr.Row):
                    ui.elem("frame_count", gr.Number, label = "Frame count", precision = 0, minimum = 1, step = 1, value = 100, groups = ["params"])
                    ui.elem("save_every_nth_frame", gr.Number, label = "Save every N-th frame", precision = 0, minimum = 1, step = 1, value = 1, groups = ["params", "session", "mode_sequence"])

                ui.elem("archive_mode", gr.Checkbox, label = "Archive mode", value = False, groups = ["params", "session", "mode_sequence"])

            with ui.elem("", gr.Accordion, label = "Initial noise", open = False):
                ui.elem("initial_noise_factor", gr.Slider, label = "Factor", minimum = 0.0, maximum = 1.0, step = 0.01, value = 0.0, groups = ["params", "session"])
                ui.elem("initial_noise_scale", gr.Slider, label = "Scale", minimum = 1, maximum = 1024, step = 1, value = 1, groups = ["params", "session"])
                ui.elem("initial_noise_octaves", gr.Slider, label = "Octaves", minimum = 1, maximum = 10, step = 1, value = 1, groups = ["params", "session"])
                ui.elem("initial_noise_lacunarity", gr.Slider, label = "Lacunarity", minimum = 0.01, maximum = 4.0, step = 0.01, value = 2.0, groups = ["params", "session"])
                ui.elem("initial_noise_persistence", gr.Slider, label = "Persistence", minimum = 0.0, maximum = 1.0, step = 0.01, value = 0.5, groups = ["params", "session"])

            with ui.elem("", gr.Accordion, label = "Processing", open = False):
                ui.elem("use_sd", gr.Checkbox, label = "Use Stable Diffusion", value = True, groups = ["params", "session"])

            with ui.elem("", gr.Accordion, label = "Multisampling", open = False):
                with ui.elem("", gr.Row):
                    ui.elem("multisampling_samples", gr.Number, label = "Sample count", precision = 0, minimum = 1, value = 1, groups = ["params", "session"])
                    ui.elem("multisampling_batch_size", gr.Number, label = "Batch size", precision = 0, minimum = 1, value = 1, groups = ["params", "session"])

                ui.elem("multisampling_trimming", gr.Slider, label = "Trimming", minimum = 0.0, maximum = 0.5, step = 0.01, value = 0.0, groups = ["params", "session"])
                ui.elem("multisampling_easing", gr.Slider, label = "Easing", minimum = 0.0, maximum = 16.0, step = 0.1, value = 0.0, groups = ["params", "session"])
                ui.elem("multisampling_preference", gr.Slider, label = "Preference", minimum = -2.0, maximum = 2.0, step = 0.1, value = 0.0, groups = ["params", "session"])

            with ui.elem("detailing_enabled", InputAccordion, label = "Detailing", value = False, groups = ["params", "session"]):
                ui.elem("detailing_scale", gr.Slider, label = "Scale", minimum = 1, maximum = 4, step = 1, value = 1, groups = ["params", "session"])
                ui.elem("detailing_scale_buffer", gr.Checkbox, label = "Scale buffer", value = False, groups = ["params", "session"])
                ui.elem("detailing_sampler", gr.Dropdown, label = "Sampling method", choices = visible_sampler_names(), value = "Euler a", groups = ["params", "session"])
                ui.elem("detailing_steps", gr.Slider, label = "Steps", minimum = 1, maximum = 150, step = 1, value = 15, groups = ["params", "session"])
                ui.elem("detailing_denoising_strength", gr.Slider, label = "Denoising strength", minimum = 0.0, maximum = 1.0, step = 0.01, value = 0.2, groups = ["params", "session"])

            with ui.elem("", gr.Accordion, label = "Frame merging", open = False):
                ui.elem("frame_merging_frames", gr.Number, label = "Frame count", precision = 0, minimum = 1, step = 1, value = 1, groups = ["params", "session"])
                ui.elem("frame_merging_trimming", gr.Slider, label = "Trimming", minimum = 0.0, maximum = 0.5, step = 0.01, value = 0.0, groups = ["params", "session"])
                ui.elem("frame_merging_easing", gr.Slider, label = "Easing", minimum = 0.0, maximum = 16.0, step = 0.1, value = 0.0, groups = ["params", "session"])
                ui.elem("frame_merging_preference", gr.Slider, label = "Preference", minimum = -2.0, maximum = 2.0, step = 0.1, value = 0.0, groups = ["params", "session"])

            with ui.elem("project_params", gr.Accordion, label = "Project", groups = ["mode_sequence"]):
                ui.elem("load_parameters", gr.Checkbox, label = "Load parameters", value = True, groups = ["params"])
                ui.elem("continue_from_last_frame", gr.Checkbox, label = "Continue from last frame", value = True, groups = ["params"])

        with ui.elem("", gr.Tab, label = "Frame Preprocessing"):
            ui.elem("preprocessing_order", gr.Dropdown, label = "Order", multiselect = True, choices = list(PREPROCESSORS.keys()), value = [], groups = ["params", "session"])

            for key, processor in PREPROCESSORS.items():
                with ui.elem(f"{key}_enabled", InputAccordion, label = processor.name, value = False, groups = ["params", "session"]):
                    with ui.elem("", gr.Row):
                        ui.elem(f"{key}_amount", gr.Slider, label = "Amount", minimum = 0.0, maximum = 1.0, step = 0.01, value = 1.0, groups = ["params", "session"])
                        ui.elem(f"{key}_amount_relative", gr.Checkbox, label = "Relative", value = False, groups = ["params", "session"])

                    with ui.elem("", gr.Tab, label = "Parameters"):
                        if processor.params:
                            for param in processor.params:
                                ui.elem(f"{key}_{param.key}", param.type, label = param.name, **param.kwargs, groups = ["params", "session"])
                        else:
                            ui.elem("", gr.Markdown, value = "_This effect has no available parameters._")

                    with ui.elem("", gr.Tab, label = "Mask"):
                        ui.elem(f"{key}_mask", gr.Pil, label = "Mask", image_mode = "L", interactive = True, groups = ["params", "session"])
                        ui.elem(f"{key}_mask_normalized", gr.Checkbox, label = "Normalized", value = False, groups = ["params", "session"])
                        ui.elem(f"{key}_mask_inverted", gr.Checkbox, label = "Inverted", value = False, groups = ["params", "session"])
                        ui.elem(f"{key}_mask_blurring", gr.Slider, label = "Blurring", minimum = 0.0, maximum = 50.0, step = 0.1, value = 0.0, groups = ["params", "session"])

        # FIXME: `Tab` cannot be hidden; an error is thrown regarding an inability to send it as an input component
        with ui.elem("", gr.Tab, label = "Video Rendering"):
            ui.elem("video_fps", gr.Slider, label = "Frames per second", minimum = 1, maximum = 60, step = 1, value = 30, groups = ["params", "mode_sequence"])
            ui.elem("video_looping", gr.Checkbox, label = "Looping", value = False, groups = ["params", "mode_sequence"])
            ui.elem("video_filtering_order", gr.Dropdown, label = "Order", multiselect = True, choices = list(FILTERS.keys()), value = [], groups = ["params", "mode_sequence"])

            for key, filter in FILTERS.items():
                # FIXME: `InputAccordion` cannot be hidden; behaves in a broken way when its visibility is changed
                with ui.elem(f"video_{key}_enabled", InputAccordion, label = filter.name, value = False, groups = ["params"]):
                    for param in filter.params:
                        ui.elem(f"video_{key}_{param.key}", param.type, label = param.name, **param.kwargs, groups = ["params", "mode_sequence"])

            with ui.elem("", gr.Row):
                ui.elem("render_draft_on_finish", gr.Checkbox, label = "Render draft when finished", value = False, groups = ["params", "mode_sequence"])
                ui.elem("render_final_on_finish", gr.Checkbox, label = "Render final when finished", value = False, groups = ["params", "mode_sequence"])

            with ui.elem("", gr.Row):
                def make_render_callback(is_final):
                    def callback(*args):
                        yield gr.update(interactive = False), gr.update(interactive = False), gr.update()

                        ext_params = ui.unpack_values(["group:params"], *args)

                        start_video_render(ext_params, is_final)
                        wait_until(lambda: not video_render_queue.busy)

                        yield gr.update(interactive = True), gr.update(interactive = True), f"{ext_params.output_dir}/{ext_params.project_subdir}-{'final' if is_final else 'draft'}.mp4"

                    return callback

                ui.elem("render_draft", gr.Button, value = "Render draft", groups = ["mode_sequence"])
                ui.callback("render_draft", "click", make_render_callback(False), ["group:params"], ["render_draft", "render_final", "video_preview"])
                ui.elem("render_final", gr.Button, value = "Render final", groups = ["mode_sequence"])
                ui.callback("render_final", "click", make_render_callback(True), ["group:params"], ["render_draft", "render_final", "video_preview"])

            ui.elem("video_preview", gr.Video, label = "Preview", format = "mp4", interactive = False, groups = ["mode_sequence"])

        with ui.elem("", gr.Tab, label = "Metrics"):
            def render_plots_callback(*args):
                ext_params = ui.unpack_values(["group:params"], *args)
                project_dir = Path(ext_params.output_dir) / ext_params.project_subdir
                metrics = Metrics()
                metrics.load(project_dir)
                return gr.update(value = metrics.plot(project_dir))

            ui.elem("metrics_enabled", gr.Checkbox, label = "Enabled", value = False, groups = ["params", "mode_sequence"])
            ui.elem("metrics_save_plots_every_nth_frame", gr.Number, label = "Save plots every N-th frame", precision = 0, minimum = 1, step = 1, value = 10, groups = ["params", "mode_sequence"])
            ui.elem("render_plots", gr.Button, value = "Render plots", groups = ["mode_sequence"])
            ui.callback("render_plots", "click", render_plots_callback, ["group:params"], ["metrics_plots"])
            ui.elem("metrics_plots", gr.Gallery, label = "Plots", columns = 4, object_fit = "contain", preview = True, groups = ["mode_sequence"])

        with ui.elem("", gr.Tab, label = "Help"):
            for file_name, title in [
                ("main.md", "Main"),
                ("tab_general.md", "General tab"),
                ("tab_frame_preprocessing.md", "Frame Preprocessing tab"),
                ("tab_video_rendering.md", "Video Rendering tab"),
                ("tab_metrics.md", "Metrics tab"),
                ("additional_notes.md", "Additional notes"),
            ]:
                with ui.elem("", gr.Accordion, label = title, open = False):
                    ui.elem("", gr.Markdown, value = load_text(EXTENSION_DIR / "docs" / "temporal" / file_name, ""))

        return ui.finalize(["group:params"])

    def run(self, p, *args):
        saved_ext_param_ids[:] = self._ui.parse_ids(["group:session"])
        ext_params = self._ui.unpack_values(["group:params"], *args)
        processed = GENERATION_MODES[ext_params.mode].func(p, ext_params)

        if ext_params.render_draft_on_finish:
            start_video_render(ext_params, False)

        if ext_params.render_final_on_finish:
            start_video_render(ext_params, True)

        return processed
