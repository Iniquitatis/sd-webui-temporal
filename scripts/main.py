from collections.abc import Iterable
from copy import copy
from inspect import isgeneratorfunction
from itertools import count
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Iterator, Optional, Type, TypeVar

import gradio as gr
from gradio.blocks import Block
from gradio.components import Component

from modules import processing, scripts, shared
from modules.options import Options
from modules.processing import Processed, StableDiffusionProcessingImg2Img
from modules.styles import StyleDatabase
from modules.shared_state import State
from modules.ui_components import ToolButton

from temporal.blend_modes import BLEND_MODES
from temporal.compat import upgrade_project
from temporal.image_filters import IMAGE_FILTERS
from temporal.interop import EXTENSION_DIR, get_cn_units
from temporal.pipeline import PIPELINE_MODULES
from temporal.preset_store import PresetStore
from temporal.project import Project, make_video_file_name, render_project_video
from temporal.project_store import ProjectStore
from temporal.session import InitialNoiseParams, Session
from temporal.ui.module_list import ModuleAccordion, ModuleList
from temporal.utils import logging
from temporal.utils.collection import get_first_element
from temporal.utils.fs import load_text
from temporal.utils.image import PILImage, generate_value_noise_image, np_to_pil, pil_to_np
from temporal.utils.object import copy_with_overrides, get_property_by_path, set_property_by_path
from temporal.utils.string import match_mask
from temporal.utils.time import wait_until
from temporal.video_filters import VIDEO_FILTERS
from temporal.video_renderer import video_render_queue
from temporal.web_ui import process_image


# FIXME: To shut up the type checker
opts: Options = getattr(shared, "opts")
prompt_styles: StyleDatabase = getattr(shared, "prompt_styles")
state: State = getattr(shared, "state")


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

        with ui.elem("", gr.Tab, label = "General"):
            with ui.elem("", gr.Accordion, label = "Output"):
                with ui.elem("", gr.Row):
                    ui.elem("output.output_dir", gr.Textbox, label = "Output directory", value = "outputs/temporal", groups = ["params"])
                    ui.elem("output.project_subdir", gr.Textbox, label = "Project subdirectory", value = "untitled", groups = ["params"])

                ui.elem("frame_count", gr.Number, label = "Frame count", precision = 0, minimum = 1, step = 1, value = 100, groups = ["params"])
                ui.elem("show_only_finalized_frames", gr.Checkbox, label = "Show only finalized frames", value = False, groups = ["params"])

            with ui.elem("", gr.Accordion, label = "Project"):
                ui.elem("load_parameters", gr.Checkbox, label = "Load parameters", value = True, groups = ["params"])
                ui.elem("continue_from_last_frame", gr.Checkbox, label = "Continue from last frame", value = True, groups = ["params"])
                ui.elem("autosave_every_n_iterations", gr.Number, label = "Autosave every N iterations", precision = 0, minimum = 1, step = 1, value = 10, groups = ["params"])

        with ui.elem("", gr.Tab, label = "Pipeline"):
            with ui.elem("", gr.Accordion, label = "Initial noise", open = False):
                ui.elem("initial_noise.factor", gr.Slider, label = "Factor", minimum = 0.0, maximum = 1.0, step = 0.01, value = 0.0, groups = ["params", "session"])
                ui.elem("initial_noise.scale", gr.Slider, label = "Scale", precision = 0, minimum = 1, maximum = 1024, step = 1, value = 1, groups = ["params", "session"])
                ui.elem("initial_noise.octaves", gr.Slider, label = "Octaves", precision = 0, minimum = 1, maximum = 10, step = 1, value = 1, groups = ["params", "session"])
                ui.elem("initial_noise.lacunarity", gr.Slider, label = "Lacunarity", minimum = 0.01, maximum = 4.0, step = 0.01, value = 2.0, groups = ["params", "session"])
                ui.elem("initial_noise.persistence", gr.Slider, label = "Persistence", minimum = 0.0, maximum = 1.0, step = 0.01, value = 0.5, groups = ["params", "session"])

            with ui.elem("pipeline.module_order", ModuleList, label = "Order", keys = PIPELINE_MODULES.keys(), groups = ["params", "session"]):
                for id, module in PIPELINE_MODULES.items():
                    with ui.elem(f"pipeline.modules['{id}'].enabled", ModuleAccordion, label = module.name, key = id, value = False, open = False, groups = ["params", "session"]):
                        ui.elem(f"pipeline.modules['{id}'].preview", gr.Checkbox, label = "Preview", value = True, groups = ["params"])

                        for param in module.__ui_params__.values():
                            ui.elem(f"pipeline.modules['{id}'].{param.key}", param.gr_type, label = param.name, **param.kwargs, groups = ["params", "session"])

        with ui.elem("", gr.Tab, label = "Image Filtering"):
            with ui.elem("image_filterer.filter_order", ModuleList, keys = IMAGE_FILTERS.keys(), groups = ["params", "session"]):
                for id, filter in IMAGE_FILTERS.items():
                    with ui.elem(f"image_filterer.filters['{id}'].enabled", ModuleAccordion, label = filter.name, key = id, value = False, open = False, groups = ["params", "session"]):
                        with ui.elem("", gr.Row):
                            ui.elem(f"image_filterer.filters['{id}'].amount", gr.Slider, label = "Amount", minimum = 0.0, maximum = 1.0, step = 0.01, value = 1.0, groups = ["params", "session"])
                            ui.elem(f"image_filterer.filters['{id}'].amount_relative", gr.Checkbox, label = "Relative", value = False, groups = ["params", "session"])

                        ui.elem(f"image_filterer.filters['{id}'].blend_mode", gr.Dropdown, label = "Blend mode", choices = list(BLEND_MODES.keys()), value = get_first_element(BLEND_MODES), groups = ["params", "session"])

                        with ui.elem("", gr.Tab, label = "Parameters"):
                            if filter.__ui_params__:
                                for param in filter.__ui_params__.values():
                                    ui.elem(f"image_filterer.filters['{id}'].{param.key}", param.gr_type, label = param.name, **param.kwargs, groups = ["params", "session"])
                            else:
                                ui.elem("", gr.Markdown, value = "_This filter has no available parameters._")

                        with ui.elem("", gr.Tab, label = "Mask"):
                            ui.elem(f"image_filterer.filters['{id}'].mask.image", gr.Image, label = "Image", type = "numpy", image_mode = "L", interactive = True, groups = ["params", "session"])
                            ui.elem(f"image_filterer.filters['{id}'].mask.normalized", gr.Checkbox, label = "Normalized", value = False, groups = ["params", "session"])
                            ui.elem(f"image_filterer.filters['{id}'].mask.inverted", gr.Checkbox, label = "Inverted", value = False, groups = ["params", "session"])
                            ui.elem(f"image_filterer.filters['{id}'].mask.blurring", gr.Slider, label = "Blurring", minimum = 0.0, maximum = 50.0, step = 0.1, value = 0.0, groups = ["params", "session"])

        with ui.elem("", gr.Tab, label = "Video Rendering"):
            ui.elem("video_renderer.fps", gr.Slider, label = "Frames per second", precision = 0, minimum = 1, maximum = 60, step = 1, value = 30, groups = ["params"])
            ui.elem("video_renderer.looping", gr.Checkbox, label = "Looping", value = False, groups = ["params"])

            with ui.elem("video_renderer.filter_order", ModuleList, keys = VIDEO_FILTERS.keys(), groups = ["params"]):
                for id, filter in VIDEO_FILTERS.items():
                    with ui.elem(f"video_renderer.filters['{id}'].enabled", ModuleAccordion, label = filter.name, key = id, value = False, open = False, groups = ["params"]):
                        for param in filter.__ui_params__.values():
                            ui.elem(f"video_renderer.filters['{id}'].{param.key}", param.gr_type, label = param.name, **param.kwargs, groups = ["params"])

            with ui.elem("", gr.Row):
                def make_render_callback(is_final):
                    def callback(inputs):
                        yield {
                            "render_draft": gr.update(interactive = False),
                            "render_final": gr.update(interactive = False),
                        }

                        session = self._ui_to_session(inputs)

                        render_project_video(session.output.output_dir, session.output.project_subdir, session.video_renderer, is_final)
                        wait_until(lambda: not video_render_queue.busy)

                        yield {
                            "render_draft": gr.update(interactive = True),
                            "render_final": gr.update(interactive = True),
                            "video_preview": (session.output.output_dir / make_video_file_name(session.output.project_subdir, is_final)).as_posix(),
                        }

                    return callback

                ui.elem("render_draft", gr.Button, value = "Render draft")
                ui.callback("render_draft", "click", make_render_callback(False), ["group:params"], ["render_draft", "render_final", "video_preview"])
                ui.elem("render_final", gr.Button, value = "Render final")
                ui.callback("render_final", "click", make_render_callback(True), ["group:params"], ["render_draft", "render_final", "video_preview"])

            ui.elem("video_preview", gr.Video, label = "Preview", format = "mp4", interactive = False)

        with ui.elem("", gr.Tab, label = "Measuring"):
            def render_plots_callback(inputs):
                session = self._ui_to_session(inputs)
                project = Project(session.output.output_dir / session.output.project_subdir)
                session.load(project.session_path)
                return {"metrics_plots": gr.update(value = list(session.pipeline.modules["measuring"].metrics.plot().values()))}

            ui.elem("render_plots", gr.Button, value = "Render plots")
            ui.callback("render_plots", "click", render_plots_callback, ["group:params"], ["metrics_plots"])
            ui.elem("metrics_plots", gr.Gallery, label = "Plots", columns = 4, object_fit = "contain", preview = True)

        with ui.elem("", gr.Tab, label = "Project Management"):
            with ui.elem("", gr.Row):
                def managed_project_callback(inputs):
                    if inputs["managed_project"] not in self._project_store.project_names:
                        return {}

                    self._project_store.path = Path(inputs["output.output_dir"])
                    project = self._project_store.open_project(inputs["managed_project"])
                    return {
                        "project_description": gr.update(value = desc if (desc := project.get_description()) else "Cannot read project data"),
                        "project_gallery": gr.update(value = project.list_all_frame_paths()[-PROJECT_GALLERY_SIZE:]),
                    }

                def refresh_projects_callback(inputs):
                    self._project_store.path = Path(inputs["output.output_dir"])
                    self._project_store.refresh_projects()
                    return {"managed_project": gr.update(choices = self._project_store.project_names)}

                def load_project_callback(inputs):
                    session = self._ui_to_session(inputs)
                    self._project_store.path = session.output.output_dir
                    project = self._project_store.open_project(inputs["managed_project"])
                    session.load(project.session_path)
                    return {k: gr.update(value = v) for k, v in self._session_to_ui(session).items()}

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

                def upgrade_project_callback(inputs):
                    self._project_store.path = Path(inputs["output.output_dir"])
                    project = self._project_store.open_project(inputs["managed_project"])
                    upgrade_project(project.path)
                    return {
                        "project_description": gr.update(value = desc if (desc := project.get_description()) else "Cannot read project data"),
                        "project_gallery": gr.update(value = project.list_all_frame_paths()[-PROJECT_GALLERY_SIZE:]),
                    }

                def delete_intermediate_frames_callback(inputs):
                    self._project_store.path = Path(inputs["output.output_dir"])
                    project = self._project_store.open_project(inputs["managed_project"])
                    project.delete_intermediate_frames()
                    return {
                        "project_description": gr.update(value = desc if (desc := project.get_description()) else "Cannot read project data"),
                        "project_gallery": gr.update(value = project.list_all_frame_paths()[-PROJECT_GALLERY_SIZE:]),
                    }

                def delete_session_data_callback(inputs):
                    self._project_store.path = Path(inputs["output.output_dir"])
                    self._project_store.open_project(inputs["managed_project"]).delete_session_data()
                    return {}

                ui.elem("upgrade_project", gr.Button, value = "Upgrade")
                ui.callback("upgrade_project", "click", upgrade_project_callback, ["output.output_dir", "managed_project"], ["project_description", "project_gallery"])
                ui.elem("delete_intermediate_frames", gr.Button, value = "Delete intermediate frames")
                ui.callback("delete_intermediate_frames", "click", delete_intermediate_frames_callback, ["output.output_dir", "managed_project"], ["project_gallery"])
                ui.elem("delete_session_data", gr.Button, value = "Delete session data")
                ui.callback("delete_session_data", "click", delete_session_data_callback, ["output.output_dir", "managed_project"], [])

        with ui.elem("", gr.Tab, label = "Help"):
            for file_name, title in [
                ("main.md", "Main"),
                ("tab_general.md", "General tab"),
                ("tab_pipeline.md", "Pipeline tab"),
                ("tab_image_filtering.md", "Image Filtering tab"),
                ("tab_video_rendering.md", "Video Rendering tab"),
                ("tab_measuring.md", "Measuring tab"),
                ("tab_project_management.md", "Project Management tab"),
                ("additional_notes.md", "Additional notes"),
            ]:
                with ui.elem("", gr.Accordion, label = title, open = False):
                    ui.elem("", gr.Markdown, value = load_text(EXTENSION_DIR / "docs" / "temporal" / file_name, ""))

        return ui.finalize(["group:params"])

    def run(self, p: StableDiffusionProcessingImg2Img, *args: Any) -> Any:
        return self._process(p, {name: arg for name, arg in zip(self._ui.parse_ids(["group:params"]), args)})

    def _ui_to_session(self, inputs: dict[str, Any], p: Optional[StableDiffusionProcessingImg2Img] = None) -> Session:
        result = Session(
            options = opts,
            processing = p,
            controlnet_units = get_cn_units(p),
        ) if p is not None else Session()

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

    def _session_to_ui(self, session: Session) -> dict[str, Any]:
        result = {}

        for id in self._ui._ids:
            try:
                data_value = get_property_by_path(session, id)
            except:
                logging.warning(f"{id} couldn't be found in {result.__class__.__name__}")

                data_value = None

            if isinstance(data_value, Path):
                result[id] = data_value.as_posix()
            else:
                result[id] = data_value

        return result

    def _process(self, p: StableDiffusionProcessingImg2Img, inputs: dict[str, Any]) -> Processed:
        opts_backup = opts.data.copy()

        opts.save_to_dirs = False

        if inputs["show_only_finalized_frames"]:
            opts.show_progress_every_n_steps = -1

        p.prompt = prompt_styles.apply_styles_to_prompt(p.prompt, p.styles)
        p.negative_prompt = prompt_styles.apply_negative_styles_to_prompt(p.negative_prompt, p.styles)
        p.styles.clear()

        processing.fix_seed(p)

        project = Project(Path(inputs["output.output_dir"]) / inputs["output.project_subdir"])
        project.load()

        if not inputs["continue_from_last_frame"]:
            project.delete_all_frames()
            project.delete_session_data()

        session = self._ui_to_session(inputs, p)

        if inputs["load_parameters"] and project.session_path.exists():
            session.load(project.session_path)

        if not self._verify_image_existence(p, session.initial_noise):
            opts.data.update(opts_backup)

            return Processed(p, p.init_images)

        last_index = project.get_last_frame_index()
        last_images = [pil_to_np(project.load_frame(last_index) or p.init_images[0])]

        state.job_count = inputs["frame_count"]

        for i, frame_index in zip(range(inputs["frame_count"]), count(last_index + 1)):
            start_time = perf_counter()

            state.job = f"Frame {frame_index + 1} / {inputs['frame_count']}"
            state.job_no = i

            if not (images := session.pipeline.run(
                session,
                last_images,
                frame_index,
                inputs["frame_count"],
                p.seed + frame_index,
                inputs["show_only_finalized_frames"],
            )) or state.interrupted or state.skipped:
                break

            if i % inputs["autosave_every_n_iterations"] == 0:
                session.save(project.session_path)
                project.save()

            last_images = images

            end_time = perf_counter()

            logging.info(f"[Temporal] Iteration took {end_time - start_time:.6f} second(s)")

        session.pipeline.finalize(session, last_images)
        session.save(project.session_path)
        project.save()

        state.end()

        opts.data.update(opts_backup)

        return Processed(p, [np_to_pil(x) for x in last_images])

    @staticmethod
    def _verify_image_existence(p: StableDiffusionProcessingImg2Img, initial_noise: InitialNoiseParams) -> bool:
        if not p.init_images or not isinstance(p.init_images[0], PILImage):
            noise = generate_value_noise_image(
                (p.width, p.height),
                3,
                initial_noise.scale,
                initial_noise.octaves,
                initial_noise.lacunarity,
                initial_noise.persistence,
                p.seed,
            )

            if initial_noise.factor < 1.0:
                if not (processed := process_image(copy_with_overrides(p,
                    init_images = [noise],
                    n_iter = 1,
                    batch_size = 1,
                    denoising_strength = 1.0 - initial_noise.factor,
                    do_not_save_samples = True,
                    do_not_save_grid = True,
                ), True)) or not processed.images:
                    return False

                p.init_images = [processed.images[0]]
            else:
                p.init_images = [noise]

        if opts.img2img_color_correction:
            p.color_corrections = [processing.setup_color_correction(p.init_images[0])]

        return True
