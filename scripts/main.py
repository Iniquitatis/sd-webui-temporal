from collections.abc import Iterable
from copy import copy
from inspect import isgeneratorfunction
from pathlib import Path
from typing import Any, Callable, Iterator, Type, TypeVar

import gradio as gr
from gradio.blocks import Block
from gradio.components import Component

from modules import scripts
from modules.processing import StableDiffusionProcessingImg2Img
from modules.sd_samplers import visible_sampler_names
from modules.ui_components import InputAccordion, ToolButton

from temporal.data import ExtensionData
from temporal.image_blending import BLEND_MODES
from temporal.image_filtering import IMAGE_FILTERS
from temporal.image_generation import GENERATION_MODES
from temporal.interop import EXTENSION_DIR
from temporal.metrics import Metrics
from temporal.preset_store import PresetStore
from temporal.project import Project, make_video_file_name
from temporal.project_store import ProjectStore
from temporal.session import Session
from temporal.ui.module_list import ModuleAccordion, ModuleList
from temporal.utils import logging
from temporal.utils.collection import get_first_element
from temporal.utils.fs import load_text
from temporal.utils.object import get_property_by_path, set_property_by_path
from temporal.utils.string import match_mask
from temporal.utils.time import wait_until
from temporal.video_filtering import VIDEO_FILTERS
from temporal.video_rendering import enqueue_video_render, video_render_queue


PROJECT_GALLERY_SIZE = 10


T = TypeVar("T", bound = Block | Component)


class UI:
    def __init__(self, id_formatter: Callable[[str], str]) -> None:
        self._id_formatter = id_formatter
        self._elems = {}
        self._ids = []
        self._groups = {}
        self._callbacks = {}
        self._existing_labels = set()

    def parse_ids(self, ids: Iterable[str]) -> list[str]:
        result = []

        for id in ids:
            if id.startswith("group:"):
                _, group = id.split(":")
                result.extend(x for x in self._ids if self.is_in_group(x, group))
            else:
                result.extend(x for x in self._ids if match_mask(x, id))

        return result

    def is_in_group(self, id: str, group: str) -> bool:
        return any(match_mask(x, group) for x in self._groups[id])

    def elem(self, id: str, constructor: Type[T], *args: Any, groups: list[str] = [], **kwargs: Any) -> T:
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

    def callback(self, id: str, event: str, func: Callable[[dict[str, Any]], dict[str, Any] | Iterator[dict[str, Any]]], inputs: Iterable[str], outputs: Iterable[str]) -> None:
        self._callbacks[id].append((event, func, inputs, outputs))

    def finalize(self, ids: Iterable[str]) -> list[Any]:
        elems = copy(self._elems)
        elem_keys = {v: k for k, v in self._elems.items()}

        def make_wrapper_func(user_func, output_keys):
            if isgeneratorfunction(user_func):
                def generator_wrapper(inputs):
                    inputs_dict = {elem_keys[k]: v for k, v in inputs.items()}

                    for outputs_dict in user_func(inputs_dict):
                        yield {elems[x]: outputs_dict.get(x, gr.update()) for x in output_keys}

                return generator_wrapper

            else:
                def normal_wrapper(inputs):
                    inputs_dict = {elem_keys[k]: v for k, v in inputs.items()}
                    outputs_dict = user_func(inputs_dict)
                    return {elems[x]: outputs_dict.get(x, gr.update()) for x in output_keys}

                return normal_wrapper

        for id, callbacks in self._callbacks.items():
            for event, func, inputs, outputs in callbacks:
                input_keys = self.parse_ids(inputs)
                output_keys = self.parse_ids(outputs)

                event_func = getattr(self._elems[id], event)
                event_func(
                    make_wrapper_func(func, output_keys),
                    inputs = {self._elems[x] for x in input_keys},
                    outputs = {self._elems[x] for x in output_keys},
                )

        result = [self._elems[x] for x in self.parse_ids(ids)]

        self._id_formatter = lambda x: x
        self._elems.clear()
        self._existing_labels.clear()

        return result


class TemporalScript(scripts.Script):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._preset_store = PresetStore(EXTENSION_DIR / "presets")
        self._preset_store.refresh_presets()
        self._project_store = ProjectStore(Path("outputs") / "temporal")
        self._project_store.refresh_projects()

    def title(self) -> str:
        return "Temporal"

    def show(self, is_img2img: bool) -> Any:
        return is_img2img

    def ui(self, is_img2img: bool) -> Any:
        self._ui = ui = UI(self.elem_id)

        with ui.elem("", gr.Row):
            def refresh_presets_callback(_):
                self._preset_store.refresh_presets()
                return {"preset": gr.update(choices = self._preset_store.preset_names)}

            def load_preset_callback(inputs):
                preset = inputs.pop("preset")
                inputs |= self._preset_store.open_preset(preset).data
                return {k: gr.update(value = v) for k, v in inputs.items()}

            def save_preset_callback(inputs):
                preset = inputs.pop("preset")
                self._preset_store.save_preset(preset, inputs)
                return {"preset": gr.update(choices = self._preset_store.preset_names, value = preset)}

            def delete_preset_callback(inputs):
                self._preset_store.delete_preset(inputs["preset"])
                return {"preset": gr.update(choices = self._preset_store.preset_names, value = get_first_element(self._preset_store.preset_names, ""))}

            ui.elem("preset", gr.Dropdown, label = "Preset", choices = self._preset_store.preset_names, allow_custom_value = True, value = get_first_element(self._preset_store.preset_names, ""))
            ui.elem("refresh_presets", ToolButton, value = "\U0001f504")
            ui.callback("refresh_presets", "click", refresh_presets_callback, [], ["preset"])
            ui.elem("load_preset", ToolButton, value = "\U0001f4c2")
            ui.callback("load_preset", "click", load_preset_callback, ["preset", "group:params"], ["group:params"])
            ui.elem("save_preset", ToolButton, value = "\U0001f4be")
            ui.callback("save_preset", "click", save_preset_callback, ["preset", "group:params"], ["preset"])
            ui.elem("delete_preset", ToolButton, value = "\U0001f5d1\ufe0f")
            ui.callback("delete_preset", "click", delete_preset_callback, ["preset"], ["preset"])

        def mode_callback(inputs):
            return {x: gr.update(visible = ui.is_in_group(x, f"mode_{inputs['mode']}")) for x in ui.parse_ids(["group:mode_*"])}

        ui.elem("mode", gr.Dropdown, label = "Mode", choices = list(GENERATION_MODES.keys()), value = "sequence", groups = ["params"])
        ui.callback("mode", "change", mode_callback, ["mode"], ["group:mode_*"])

        with ui.elem("", gr.Tab, label = "General"):
            with ui.elem("", gr.Accordion, label = "Output"):
                with ui.elem("", gr.Row):
                    ui.elem("output.output_dir", gr.Textbox, label = "Output directory", value = "outputs/temporal", groups = ["params"])
                    ui.elem("output.project_subdir", gr.Textbox, label = "Project subdirectory", value = "untitled", groups = ["params", "mode_sequence"])

                with ui.elem("", gr.Row):
                    ui.elem("output.frame_count", gr.Number, label = "Frame count", precision = 0, minimum = 1, step = 1, value = 100, groups = ["params"])
                    ui.elem("output.save_every_nth_frame", gr.Number, label = "Save every N-th frame", precision = 0, minimum = 1, step = 1, value = 1, groups = ["params", "session", "mode_sequence"])

                ui.elem("output.archive_mode", gr.Checkbox, label = "Archive mode", value = False, groups = ["params", "session", "mode_sequence"])

            with ui.elem("", gr.Accordion, label = "Initial noise", open = False):
                ui.elem("initial_noise.factor", gr.Slider, label = "Factor", minimum = 0.0, maximum = 1.0, step = 0.01, value = 0.0, groups = ["params", "session"])
                ui.elem("initial_noise.scale", gr.Slider, label = "Scale", minimum = 1, maximum = 1024, step = 1, value = 1, groups = ["params", "session"])
                ui.elem("initial_noise.octaves", gr.Slider, label = "Octaves", minimum = 1, maximum = 10, step = 1, value = 1, groups = ["params", "session"])
                ui.elem("initial_noise.lacunarity", gr.Slider, label = "Lacunarity", minimum = 0.01, maximum = 4.0, step = 0.01, value = 2.0, groups = ["params", "session"])
                ui.elem("initial_noise.persistence", gr.Slider, label = "Persistence", minimum = 0.0, maximum = 1.0, step = 0.01, value = 0.5, groups = ["params", "session"])

            with ui.elem("", gr.Accordion, label = "Processing", open = False):
                ui.elem("processing.use_sd", gr.Checkbox, label = "Use Stable Diffusion", value = True, groups = ["params", "session"])
                ui.elem("processing.show_only_finalized_frames", gr.Checkbox, label = "Show only finalized frames", value = False, groups = ["params"])

            with ui.elem("", gr.Accordion, label = "Multisampling", open = False):
                with ui.elem("", gr.Row):
                    ui.elem("multisampling.samples", gr.Number, label = "Sample count", precision = 0, minimum = 1, value = 1, groups = ["params", "session"])
                    ui.elem("multisampling.batch_size", gr.Number, label = "Batch size", precision = 0, minimum = 1, value = 1, groups = ["params", "session"])

                ui.elem("multisampling.trimming", gr.Slider, label = "Trimming", minimum = 0.0, maximum = 0.5, step = 0.01, value = 0.0, groups = ["params", "session"])
                ui.elem("multisampling.easing", gr.Slider, label = "Easing", minimum = 0.0, maximum = 16.0, step = 0.1, value = 0.0, groups = ["params", "session"])
                ui.elem("multisampling.preference", gr.Slider, label = "Preference", minimum = -2.0, maximum = 2.0, step = 0.1, value = 0.0, groups = ["params", "session"])

            with ui.elem("detailing.enabled", InputAccordion, label = "Detailing", value = False, groups = ["params", "session"]):
                ui.elem("detailing.scale", gr.Slider, label = "Scale", minimum = 1.0, maximum = 4.0, step = 0.25, value = 1.0, groups = ["params", "session"])
                ui.elem("detailing.scale_buffer", gr.Checkbox, label = "Scale buffer", value = False, groups = ["params", "session"])
                ui.elem("detailing.sampler", gr.Dropdown, label = "Sampling method", choices = visible_sampler_names(), value = "Euler a", groups = ["params", "session"])
                ui.elem("detailing.steps", gr.Slider, label = "Steps", minimum = 1, maximum = 150, step = 1, value = 15, groups = ["params", "session"])
                ui.elem("detailing.denoising_strength", gr.Slider, label = "Denoising strength", minimum = 0.0, maximum = 1.0, step = 0.01, value = 0.2, groups = ["params", "session"])

            with ui.elem("", gr.Accordion, label = "Frame merging", open = False):
                ui.elem("frame_merging.frames", gr.Number, label = "Frame count", precision = 0, minimum = 1, step = 1, value = 1, groups = ["params", "session"])
                ui.elem("frame_merging.trimming", gr.Slider, label = "Trimming", minimum = 0.0, maximum = 0.5, step = 0.01, value = 0.0, groups = ["params", "session"])
                ui.elem("frame_merging.easing", gr.Slider, label = "Easing", minimum = 0.0, maximum = 16.0, step = 0.1, value = 0.0, groups = ["params", "session"])
                ui.elem("frame_merging.preference", gr.Slider, label = "Preference", minimum = -2.0, maximum = 2.0, step = 0.1, value = 0.0, groups = ["params", "session"])

            with ui.elem("project_params", gr.Accordion, label = "Project", groups = ["mode_sequence"]):
                ui.elem("project.load_parameters", gr.Checkbox, label = "Load parameters", value = True, groups = ["params"])
                ui.elem("project.continue_from_last_frame", gr.Checkbox, label = "Continue from last frame", value = True, groups = ["params"])

        with ui.elem("", gr.Tab, label = "Image Filtering"):
            with ui.elem("filtering.filter_order", ModuleList, keys = IMAGE_FILTERS.keys(), groups = ["params", "session"]):
                for id, filter in IMAGE_FILTERS.items():
                    with ui.elem(f"filtering.filter_data['{id}'].enabled", ModuleAccordion, label = filter.name, key = id, value = False, open = False, groups = ["params", "session"]):
                        with ui.elem("", gr.Row):
                            ui.elem(f"filtering.filter_data['{id}'].amount", gr.Slider, label = "Amount", minimum = 0.0, maximum = 1.0, step = 0.01, value = 1.0, groups = ["params", "session"])
                            ui.elem(f"filtering.filter_data['{id}'].amount_relative", gr.Checkbox, label = "Relative", value = False, groups = ["params", "session"])

                        ui.elem(f"filtering.filter_data['{id}'].blend_mode", gr.Dropdown, label = "Blend mode", choices = list(BLEND_MODES.keys()), value = get_first_element(BLEND_MODES), groups = ["params", "session"])

                        with ui.elem("", gr.Tab, label = "Parameters"):
                            if filter.params:
                                for param in filter.params.values():
                                    ui.elem(f"filtering.filter_data['{id}'].params.{param.id}", param.type, label = param.name, **param.kwargs, groups = ["params", "session"])
                            else:
                                ui.elem("", gr.Markdown, value = "_This effect has no available parameters._")

                        with ui.elem("", gr.Tab, label = "Mask"):
                            ui.elem(f"filtering.filter_data['{id}'].mask.image", gr.Pil, label = "Image", image_mode = "L", interactive = True, groups = ["params", "session"])
                            ui.elem(f"filtering.filter_data['{id}'].mask.normalized", gr.Checkbox, label = "Normalized", value = False, groups = ["params", "session"])
                            ui.elem(f"filtering.filter_data['{id}'].mask.inverted", gr.Checkbox, label = "Inverted", value = False, groups = ["params", "session"])
                            ui.elem(f"filtering.filter_data['{id}'].mask.blurring", gr.Slider, label = "Blurring", minimum = 0.0, maximum = 50.0, step = 0.1, value = 0.0, groups = ["params", "session"])

        # FIXME: `Tab` cannot be hidden; an error is thrown regarding an inability to send it as an input component
        with ui.elem("", gr.Tab, label = "Video Rendering"):
            ui.elem("video.fps", gr.Slider, label = "Frames per second", minimum = 1, maximum = 60, step = 1, value = 30, groups = ["params", "mode_sequence"])
            ui.elem("video.looping", gr.Checkbox, label = "Looping", value = False, groups = ["params", "mode_sequence"])

            with ui.elem("video.filter_order", ModuleList, keys = VIDEO_FILTERS.keys(), groups = ["params", "mode_sequence"]):
                for id, filter in VIDEO_FILTERS.items():
                    with ui.elem(f"video.filter_data['{id}'].enabled", ModuleAccordion, label = filter.name, key = id, value = False, open = False, groups = ["params"]):
                        for param in filter.params.values():
                            ui.elem(f"video.filter_data['{id}'].params.{param.id}", param.type, label = param.name, **param.kwargs, groups = ["params", "mode_sequence"])

            with ui.elem("", gr.Row):
                ui.elem("render_draft_on_finish", gr.Checkbox, label = "Render draft when finished", value = False, groups = ["params", "mode_sequence"])
                ui.elem("render_final_on_finish", gr.Checkbox, label = "Render final when finished", value = False, groups = ["params", "mode_sequence"])

            with ui.elem("", gr.Row):
                def make_render_callback(is_final):
                    def callback(inputs):
                        yield {
                            "render_draft": gr.update(interactive = False),
                            "render_final": gr.update(interactive = False),
                        }

                        data = self._ui_to_ext_data(inputs)

                        self._start_video_render(data, is_final)
                        wait_until(lambda: not video_render_queue.busy)

                        yield {
                            "render_draft": gr.update(interactive = True),
                            "render_final": gr.update(interactive = True),
                            "video_preview": f"{data.output.output_dir}/{make_video_file_name(data.output.project_subdir, is_final)}",
                        }

                    return callback

                ui.elem("render_draft", gr.Button, value = "Render draft", groups = ["mode_sequence"])
                ui.callback("render_draft", "click", make_render_callback(False), ["group:params"], ["render_draft", "render_final", "video_preview"])
                ui.elem("render_final", gr.Button, value = "Render final", groups = ["mode_sequence"])
                ui.callback("render_final", "click", make_render_callback(True), ["group:params"], ["render_draft", "render_final", "video_preview"])

            ui.elem("video_preview", gr.Video, label = "Preview", format = "mp4", interactive = False, groups = ["mode_sequence"])

        with ui.elem("", gr.Tab, label = "Metrics"):
            def render_plots_callback(inputs):
                data = self._ui_to_ext_data(inputs)
                project = Project(data.output.output_dir / data.output.project_subdir)
                metrics = Metrics()
                metrics.load(project.metrics_path)
                return {"metrics_plots": gr.update(value = list(metrics.plot().values()))}

            ui.elem("measuring.enabled", gr.Checkbox, label = "Enabled", value = False, groups = ["params", "mode_sequence"])
            ui.elem("measuring.plot_every_nth_frame", gr.Number, label = "Save plots every N-th frame", precision = 0, minimum = 1, step = 1, value = 10, groups = ["params", "mode_sequence"])
            ui.elem("render_plots", gr.Button, value = "Render plots", groups = ["mode_sequence"])
            ui.callback("render_plots", "click", render_plots_callback, ["group:params"], ["metrics_plots"])
            ui.elem("metrics_plots", gr.Gallery, label = "Plots", columns = 4, object_fit = "contain", preview = True, groups = ["mode_sequence"])

        with ui.elem("", gr.Tab, label = "Project Management"):
            with ui.elem("", gr.Row):
                def managed_project_callback(inputs):
                    if inputs["managed_project"] not in self._project_store.project_names:
                        return {}

                    self._project_store.path = Path(inputs["output.output_dir"])
                    project = self._project_store.open_project(inputs["managed_project"])
                    return {
                        "project_description": gr.update(value = project.get_description()),
                        "project_gallery": gr.update(value = project.list_all_frame_paths()[-PROJECT_GALLERY_SIZE:]),
                    }

                def refresh_projects_callback(inputs):
                    self._project_store.path = Path(inputs["output.output_dir"])
                    self._project_store.refresh_projects()
                    return {"managed_project": gr.update(choices = self._project_store.project_names)}

                def load_project_callback(inputs):
                    data = self._ui_to_ext_data(inputs)
                    self._project_store.path = Path(data.output.output_dir)
                    project = self._project_store.open_project(inputs["managed_project"])
                    session = Session(ext_data = data)
                    session.load(project.session_path)
                    return {k: gr.update(value = v) for k, v in self._ext_data_to_ui(data).items()}

                def delete_project_callback(inputs):
                    self._project_store.path = Path(inputs["output.output_dir"])
                    self._project_store.delete_project(inputs["managed_project"])
                    return {"managed_project": gr.update(choices = self._project_store.project_names, value = get_first_element(self._project_store.project_names, ""))}

                ui.elem("managed_project", gr.Dropdown, label = "Project", choices = self._project_store.project_names, allow_custom_value = True, value = get_first_element(self._project_store.project_names, ""))
                # FIXME: `change` makes typing slower, but `select` won't work until user clicks an appropriate item
                ui.callback("managed_project", "change", managed_project_callback, ["output.output_dir", "managed_project"], ["project_description", "project_gallery"])
                ui.elem("refresh_projects", ToolButton, value = "\U0001f504")
                ui.callback("refresh_projects", "click", refresh_projects_callback, ["output.output_dir"], ["managed_project"])
                ui.elem("load_project", ToolButton, value = "\U0001f4c2")
                ui.callback("load_project", "click", load_project_callback, ["output.output_dir", "managed_project", "group:session"], ["group:session"])
                ui.elem("delete_project", ToolButton, value = "\U0001f5d1\ufe0f")
                ui.callback("delete_project", "click", delete_project_callback, ["output.output_dir", "managed_project"], ["managed_project"])

            with ui.elem("", gr.Accordion, label = "Information", open = False):
                ui.elem("project_description", gr.Textbox, label = "Description", lines = 5, max_lines = 5, interactive = False)
                ui.elem("project_gallery", gr.Gallery, label = f"Last {PROJECT_GALLERY_SIZE} images", columns = 4, object_fit = "contain", preview = True)

            with ui.elem("", gr.Accordion, label = "Tools", open = False):
                with ui.elem("", gr.Row):
                    def rename_project_callback(inputs):
                        self._project_store.path = Path(inputs["output.output_dir"])
                        self._project_store.rename_project(inputs["managed_project"], inputs["new_project_name"])
                        return {"managed_project": gr.update(choices = self._project_store.project_names, value = inputs["new_project_name"])}

                    ui.elem("new_project_name", gr.Textbox, label = "New name", value = "")
                    ui.elem("confirm_project_rename", ToolButton, value = "\U00002714\ufe0f")
                    ui.callback("confirm_project_rename", "click", rename_project_callback, ["output.output_dir", "managed_project", "new_project_name"], ["managed_project"])

                def delete_intermediate_frames_callback(inputs):
                    self._project_store.path = Path(inputs["output.output_dir"])
                    self._project_store.open_project(inputs["managed_project"]).delete_intermediate_frames()
                    return {}

                def delete_session_data_callback(inputs):
                    self._project_store.path = Path(inputs["output.output_dir"])
                    self._project_store.open_project(inputs["managed_project"]).delete_session_data()
                    return {}

                ui.elem("delete_intermediate_frames", gr.Button, value = "Delete intermediate frames")
                ui.callback("delete_intermediate_frames", "click", delete_intermediate_frames_callback, ["output.output_dir", "managed_project"], [])
                ui.elem("delete_session_data", gr.Button, value = "Delete session data")
                ui.callback("delete_session_data", "click", delete_session_data_callback, ["output.output_dir", "managed_project"], [])

        with ui.elem("", gr.Tab, label = "Help"):
            for file_name, title in [
                ("main.md", "Main"),
                ("tab_general.md", "General tab"),
                ("tab_frame_preprocessing.md", "Frame Preprocessing tab"),
                ("tab_video_rendering.md", "Video Rendering tab"),
                ("tab_metrics.md", "Metrics tab"),
                ("tab_project_management.md", "Project Management tab"),
                ("additional_notes.md", "Additional notes"),
            ]:
                with ui.elem("", gr.Accordion, label = title, open = False):
                    ui.elem("", gr.Markdown, value = load_text(EXTENSION_DIR / "docs" / "temporal" / file_name, ""))

        return ui.finalize(["group:params"])

    def run(self, p: StableDiffusionProcessingImg2Img, *args: Any) -> Any:
        data = self._ui_to_ext_data({name: arg for name, arg in zip(self._ui.parse_ids(["group:params"]), args)})
        processed = GENERATION_MODES[data.mode].process(p, data)

        # FIXME: Feels somewhat hacky
        if data.mode == "sequence":
            if data.render_draft_on_finish:
                self._start_video_render(data, False)

            if data.render_final_on_finish:
                self._start_video_render(data, True)

        return processed

    def _start_video_render(self, data: ExtensionData, is_final: bool) -> None:
        output_dir = Path(data.output.output_dir)
        project = Project(output_dir / data.output.project_subdir)
        enqueue_video_render(output_dir / make_video_file_name(project.name, is_final), project.list_all_frame_paths(), data.video, is_final)

    def _ui_to_ext_data(self, inputs: dict[str, Any]) -> ExtensionData:
        result = ExtensionData()

        for key, ui_value in inputs.items():
            try:
                data_value = get_property_by_path(result, key)
            except:
                logging.warning(f"{key} couldn't be found in {result.__class__.__name__}")

                data_value = None

            if isinstance(data_value, Path):
                set_property_by_path(result, key, Path(ui_value))
            else:
                set_property_by_path(result, key, ui_value)

        return result

    def _ext_data_to_ui(self, ext_data: ExtensionData) -> dict[str, Any]:
        result = {}

        for id in self._ui._ids:
            try:
                data_value = get_property_by_path(ext_data, id)
            except:
                logging.warning(f"{id} couldn't be found in {result.__class__.__name__}")

                data_value = None

            if isinstance(data_value, Path):
                result[id] = data_value.as_posix()
            else:
                result[id] = data_value

        return result
