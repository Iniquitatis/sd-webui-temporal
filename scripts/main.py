from collections.abc import Iterable
from copy import copy
from inspect import isgeneratorfunction
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
from temporal.global_options import OptionCategory, global_options
from temporal.image_filters import ImageFilter
from temporal.interop import EXTENSION_DIR, get_cn_units
from temporal.pipeline import PIPELINE_MODULES
from temporal.preset_store import PresetStore
from temporal.project import Project, render_project_video
from temporal.project_store import ProjectStore
from temporal.session import InitialNoiseParams, Session
from temporal.ui.module_list import ModuleAccordion, ModuleList
from temporal.utils import logging
from temporal.utils.collection import get_first_element
from temporal.utils.fs import load_text
from temporal.utils.image import PILImage, np_to_pil, pil_to_np
from temporal.utils.numpy import generate_value_noise
from temporal.utils.object import copy_with_overrides, get_property_by_path, set_property_by_path
from temporal.utils.string import match_mask
from temporal.utils.time import wait_until
from temporal.video_filters import VIDEO_FILTERS
from temporal.video_renderer import video_render_queue
from temporal.web_ui import process_images


# FIXME: To shut up the type checker
opts: Options = getattr(shared, "opts")
prompt_styles: StyleDatabase = getattr(shared, "prompt_styles")
state: State = getattr(shared, "state")


CallbackFunc = Callable[[dict[str, Any]], dict[str, Any] | Iterator[dict[str, Any]]]

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

    def callback(self, id: str, event: str, inputs: Iterable[str], outputs: Iterable[str]) -> Callable[[CallbackFunc], CallbackFunc]:
        def decorator(func: CallbackFunc) -> CallbackFunc:
            self._callbacks[id].append((event, func, inputs, outputs))
            return func

        return decorator

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
        global_options.load(EXTENSION_DIR / "settings")
        self._preset_store = PresetStore(EXTENSION_DIR / "presets")
        self._preset_store.refresh_presets()
        self._project_store = ProjectStore(Path(global_options.output.output_dir))
        self._project_store.refresh_projects()

    def title(self) -> str:
        return "Temporal"

    def show(self, is_img2img: bool) -> Any:
        return is_img2img

    def ui(self, is_img2img: bool) -> Any:
        self._ui = ui = UI(self.elem_id)

        with ui.elem("", gr.Row):
            ui.elem("preset", gr.Dropdown, label = "Preset", choices = self._preset_store.preset_names, allow_custom_value = True, value = get_first_element(self._preset_store.preset_names, ""))
            ui.elem("refresh_presets", ToolButton, value = "\U0001f504")
            ui.elem("load_preset", ToolButton, value = "\U0001f4c2")
            ui.elem("save_preset", ToolButton, value = "\U0001f4be")
            ui.elem("delete_preset", ToolButton, value = "\U0001f5d1\ufe0f")

            @ui.callback("refresh_presets", "click", [], ["preset"])
            def _(_):
                self._preset_store.refresh_presets()
                return {"preset": gr.update(choices = self._preset_store.preset_names)}

            @ui.callback("load_preset", "click", ["preset", "group:params"], ["group:params"])
            def _(inputs):
                preset = inputs.pop("preset")
                inputs |= self._preset_store.open_preset(preset).data
                return {k: gr.update(value = v) for k, v in inputs.items()}

            @ui.callback("save_preset", "click", ["preset", "group:params"], ["preset"])
            def _(inputs):
                preset = inputs.pop("preset")
                self._preset_store.save_preset(preset, inputs)
                return {"preset": gr.update(choices = self._preset_store.preset_names, value = preset)}

            @ui.callback("delete_preset", "click", ["preset"], ["preset"])
            def _(inputs):
                self._preset_store.delete_preset(inputs["preset"])
                return {"preset": gr.update(choices = self._preset_store.preset_names, value = get_first_element(self._preset_store.preset_names, ""))}

        with ui.elem("", gr.Tab, label = "Project"):
            with ui.elem("", gr.Row):
                ui.elem("project_name", gr.Dropdown, label = "Project name", choices = self._project_store.project_names, allow_custom_value = True, value = get_first_element(self._project_store.project_names, ""), groups = ["params"])
                ui.elem("refresh_projects", ToolButton, value = "\U0001f504")
                ui.elem("load_project", ToolButton, value = "\U0001f4c2")
                ui.elem("delete_project", ToolButton, value = "\U0001f5d1\ufe0f")

                # FIXME: `change` makes typing slower, but `select` won't work until user clicks an appropriate item
                @ui.callback("project_name", "change", ["project_name"], ["project_description", "project_gallery", "project_gallery_parallel_index"])
                def _(inputs):
                    if inputs["project_name"] not in self._project_store.project_names:
                        return {}

                    self._project_store.path = Path(global_options.output.output_dir)
                    project = self._project_store.open_project(inputs["project_name"])
                    return {
                        "project_description": gr.update(value = desc if (desc := project.get_description()) else "Cannot read project data"),
                        "project_gallery": gr.update(value = project.list_all_frame_paths()[-global_options.project_management.gallery_size:]),
                        "project_gallery_parallel_index": gr.update(value = 1),
                    }

                @ui.callback("refresh_projects", "click", [], ["project_name"])
                def _(_):
                    self._project_store.path = Path(global_options.output.output_dir)
                    self._project_store.refresh_projects()
                    return {"project_name": gr.update(choices = self._project_store.project_names)}

                @ui.callback("load_project", "click", ["project_name", "group:session"], ["group:session"])
                def _(inputs):
                    self._project_store.path = Path(global_options.output.output_dir)
                    project = self._project_store.open_project(inputs["project_name"])
                    session = self._ui_to_session(inputs)
                    session.load(project.session_path)
                    return {k: gr.update(value = v) for k, v in self._session_to_ui(session).items()}

                @ui.callback("delete_project", "click", ["project_name"], ["project_name"])
                def _(inputs):
                    self._project_store.path = Path(global_options.output.output_dir)
                    self._project_store.delete_project(inputs["project_name"])
                    return {"project_name": gr.update(choices = self._project_store.project_names, value = get_first_element(self._project_store.project_names, ""))}

            with ui.elem("", gr.Tab, label = "Session"):
                ui.elem("load_parameters", gr.Checkbox, label = "Load parameters", value = True, groups = ["params"])
                ui.elem("continue_from_last_frame", gr.Checkbox, label = "Continue from last frame", value = True, groups = ["params"])
                ui.elem("iter_count", gr.Number, label = "Iteration count", precision = 0, minimum = 1, step = 1, value = 100, groups = ["params"])

            with ui.elem("", gr.Tab, label = "Information"):
                ui.elem("project_description", gr.Textbox, label = "Description", lines = 5, max_lines = 5, interactive = False)
                ui.elem("project_gallery", gr.Gallery, label = "Last images", columns = 4, object_fit = "contain", preview = True)
                ui.elem("project_gallery_parallel_index", gr.Number, label = "Parallel index", precision = 0, minimum = 1, step = 1, value = 1)

                @ui.callback("project_gallery_parallel_index", "change", ["project_name", "project_gallery_parallel_index"], ["project_gallery"])
                def _(inputs):
                    self._project_store.path = Path(global_options.output.output_dir)
                    project = self._project_store.open_project(inputs["project_name"])
                    return {"project_gallery": gr.update(value = project.list_all_frame_paths(inputs["project_gallery_parallel_index"])[-global_options.project_management.gallery_size:])}

            with ui.elem("", gr.Tab, label = "Tools"):
                with ui.elem("", gr.Row):
                    ui.elem("new_project_name", gr.Textbox, label = "New name", value = "")
                    ui.elem("confirm_project_rename", ToolButton, value = "\U00002714\ufe0f")

                    @ui.callback("confirm_project_rename", "click", ["project_name", "new_project_name"], ["project_name"])
                    def _(inputs):
                        self._project_store.path = Path(global_options.output.output_dir)
                        self._project_store.rename_project(inputs["project_name"], inputs["new_project_name"])
                        return {"project_name": gr.update(choices = self._project_store.project_names, value = inputs["new_project_name"])}

                ui.elem("upgrade_project", gr.Button, value = "Upgrade")
                ui.elem("delete_intermediate_frames", gr.Button, value = "Delete intermediate frames")
                ui.elem("delete_session_data", gr.Button, value = "Delete session data")

                @ui.callback("upgrade_project", "click", ["project_name"], ["project_description", "project_gallery"])
                def _(inputs):
                    self._project_store.path = Path(global_options.output.output_dir)
                    project = self._project_store.open_project(inputs["project_name"])
                    upgrade_project(project.path)
                    return {
                        "project_description": gr.update(value = desc if (desc := project.get_description()) else "Cannot read project data"),
                        "project_gallery": gr.update(value = project.list_all_frame_paths()[-global_options.project_management.gallery_size:]),
                    }

                @ui.callback("delete_intermediate_frames", "click", ["project_name"], ["project_gallery"])
                def _(inputs):
                    self._project_store.path = Path(global_options.output.output_dir)
                    project = self._project_store.open_project(inputs["project_name"])
                    project.delete_intermediate_frames()
                    return {
                        "project_description": gr.update(value = desc if (desc := project.get_description()) else "Cannot read project data"),
                        "project_gallery": gr.update(value = project.list_all_frame_paths()[-global_options.project_management.gallery_size:]),
                    }

                @ui.callback("delete_session_data", "click", ["project_name"], [])
                def _(inputs):
                    self._project_store.path = Path(global_options.output.output_dir)
                    self._project_store.open_project(inputs["project_name"]).delete_session_data()
                    return {}

        with ui.elem("", gr.Tab, label = "Pipeline"):
            with ui.elem("", gr.Accordion, label = "Initial noise", open = False):
                ui.elem("initial_noise.factor", gr.Slider, label = "Factor", minimum = 0.0, maximum = 1.0, step = 0.01, value = 0.0, groups = ["params", "session"])
                ui.elem("initial_noise.scale", gr.Slider, label = "Scale", precision = 0, minimum = 1, maximum = 1024, step = 1, value = 1, groups = ["params", "session"])
                ui.elem("initial_noise.octaves", gr.Slider, label = "Octaves", precision = 0, minimum = 1, maximum = 10, step = 1, value = 1, groups = ["params", "session"])
                ui.elem("initial_noise.lacunarity", gr.Slider, label = "Lacunarity", minimum = 0.01, maximum = 4.0, step = 0.01, value = 2.0, groups = ["params", "session"])
                ui.elem("initial_noise.persistence", gr.Slider, label = "Persistence", minimum = 0.0, maximum = 1.0, step = 0.01, value = 0.5, groups = ["params", "session"])

            sorted_modules = dict(sorted(PIPELINE_MODULES.items(), key = lambda x: f"{x[1].icon} {x[1].id}"))

            ui.elem("pipeline.parallel", gr.Number, label = "Parallel", precision = 0, minimum = 1, step = 1, value = 1, groups = ["params", "session"])

            with ui.elem("pipeline.module_order", ModuleList, keys = sorted_modules.keys(), groups = ["params", "session"]):
                for id, module in sorted_modules.items():
                    with ui.elem(f"pipeline.modules['{id}'].enabled", ModuleAccordion, label = f"{module.icon} {module.name}", key = id, value = False, open = False, groups = ["params", "session"]):
                        ui.elem(f"pipeline.modules['{id}'].preview", gr.Checkbox, label = "Preview", value = True, groups = ["params", "session"])

                        if issubclass(module, ImageFilter):
                            with ui.elem("", gr.Row):
                                ui.elem(f"pipeline.modules['{id}'].amount", gr.Slider, label = "Amount", minimum = 0.0, maximum = 1.0, step = 0.01, value = 1.0, groups = ["params", "session"])
                                ui.elem(f"pipeline.modules['{id}'].amount_relative", gr.Checkbox, label = "Relative", value = False, groups = ["params", "session"])

                            ui.elem(f"pipeline.modules['{id}'].blend_mode", gr.Dropdown, label = "Blend mode", choices = list(BLEND_MODES.keys()), value = get_first_element(BLEND_MODES), groups = ["params", "session"])

                            with ui.elem("", gr.Tab, label = "Parameters"):
                                if module.__ui_params__:
                                    for param in module.__ui_params__.values():
                                        ui.elem(f"pipeline.modules['{id}'].{param.key}", param.gr_type, label = param.name, **param.kwargs, groups = ["params", "session"])
                                else:
                                    ui.elem("", gr.Markdown, value = "_This filter has no available parameters._")

                            with ui.elem("", gr.Tab, label = "Mask"):
                                ui.elem(f"pipeline.modules['{id}'].mask.image", gr.Image, label = "Image", type = "numpy", image_mode = "L", interactive = True, groups = ["params", "session"])
                                ui.elem(f"pipeline.modules['{id}'].mask.normalized", gr.Checkbox, label = "Normalized", value = False, groups = ["params", "session"])
                                ui.elem(f"pipeline.modules['{id}'].mask.inverted", gr.Checkbox, label = "Inverted", value = False, groups = ["params", "session"])
                                ui.elem(f"pipeline.modules['{id}'].mask.blurring", gr.Slider, label = "Blurring", minimum = 0.0, maximum = 50.0, step = 0.1, value = 0.0, groups = ["params", "session"])

                        else:
                            for param in module.__ui_params__.values():
                                ui.elem(f"pipeline.modules['{id}'].{param.key}", param.gr_type, label = param.name, **param.kwargs, groups = ["params", "session"])

        with ui.elem("", gr.Tab, label = "Video Rendering"):
            ui.elem("video_renderer.fps", gr.Slider, label = "Frames per second", precision = 0, minimum = 1, maximum = 60, step = 1, value = 30, groups = ["params"])
            ui.elem("video_renderer.looping", gr.Checkbox, label = "Looping", value = False, groups = ["params"])

            with ui.elem("video_renderer.filter_order", ModuleList, keys = VIDEO_FILTERS.keys(), groups = ["params"]):
                for id, filter in VIDEO_FILTERS.items():
                    with ui.elem(f"video_renderer.filters['{id}'].enabled", ModuleAccordion, label = filter.name, key = id, value = False, open = False, groups = ["params"]):
                        for param in filter.__ui_params__.values():
                            ui.elem(f"video_renderer.filters['{id}'].{param.key}", param.gr_type, label = param.name, **param.kwargs, groups = ["params"])

            ui.elem("video_parallel_index", gr.Number, label = "Parallel index", precision = 0, minimum = 1, step = 1, value = 1, groups = ["params"])

            with ui.elem("", gr.Row):
                ui.elem("render_draft", gr.Button, value = "Render draft")
                ui.elem("render_final", gr.Button, value = "Render final")

                def render_video(inputs, is_final):
                    yield {
                        "render_draft": gr.update(interactive = False),
                        "render_final": gr.update(interactive = False),
                    }

                    session = self._ui_to_session(inputs)

                    video_path = render_project_video(
                        Path(global_options.output.output_dir) / session.project_name,
                        session.video_renderer,
                        is_final,
                        inputs["video_parallel_index"],
                    )
                    wait_until(lambda: not video_render_queue.busy)

                    yield {
                        "render_draft": gr.update(interactive = True),
                        "render_final": gr.update(interactive = True),
                        "video_preview": video_path.as_posix(),
                    }

                @ui.callback("render_draft", "click", ["group:params"], ["render_draft", "render_final", "video_preview"])
                def _(inputs):
                    yield from render_video(inputs, False)

                @ui.callback("render_final", "click", ["group:params"], ["render_draft", "render_final", "video_preview"])
                def _(inputs):
                    yield from render_video(inputs, True)

            ui.elem("video_preview", gr.Video, label = "Preview", format = "mp4", interactive = False)

        with ui.elem("", gr.Tab, label = "Measuring"):
            ui.elem("render_plots", gr.Button, value = "Render plots")
            ui.elem("metrics_plots", gr.Gallery, label = "Plots", columns = 4, object_fit = "contain", preview = True)

            @ui.callback("render_plots", "click", ["group:params"], ["metrics_plots"])
            def _(inputs):
                session = self._ui_to_session(inputs)
                project = Project(Path(global_options.output.output_dir) / session.project_name)
                session.load(project.session_path)
                return {"metrics_plots": gr.update(value = list(session.pipeline.modules["measuring"].metrics.plot().values()))}

        with ui.elem("", gr.Tab, label = "Settings"):
            ui.elem("apply_settings", gr.Button, value = "Apply")

            @ui.callback("apply_settings", "click", ["group:options"], [])
            def _(inputs):
                for key, ui_value in inputs.items():
                    set_property_by_path(global_options, key, ui_value)

                global_options.save(EXTENSION_DIR / "settings")

                return {}

            for field in global_options.__fields__.values():
                if not isinstance(category := getattr(global_options, field.key), OptionCategory):
                    continue

                with ui.elem("", gr.Accordion, label = category.name, open = False):
                    def make_param_getter(category: OptionCategory, key: str) -> Callable[[], Any]:
                        return lambda: getattr(category, key)

                    for param in category.__ui_params__.values():
                        kwargs = dict(param.kwargs)
                        kwargs.pop("value")

                        ui.elem(f"{field.key}.{param.key}", param.gr_type, label = param.name, value = make_param_getter(category, param.key), **kwargs, groups = ["options"])

        with ui.elem("", gr.Tab, label = "Help"):
            for file_name, title in [
                ("main.md", "Main"),
                ("tab_project.md", "Project tab"),
                ("tab_pipeline.md", "Pipeline tab"),
                ("tab_video_rendering.md", "Video Rendering tab"),
                ("tab_measuring.md", "Measuring tab"),
                ("tab_settings.md", "Settings tab"),
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
                get_property_by_path(result, key)
            except:
                logging.warning(f"{key} couldn't be found in {result.__class__.__name__}")

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

            result[id] = data_value

        return result

    def _process(self, p: StableDiffusionProcessingImg2Img, inputs: dict[str, Any]) -> Processed:
        opts_backup = opts.data.copy()

        opts.save_to_dirs = False

        if global_options.live_preview.show_only_finished_images:
            opts.show_progress_every_n_steps = -1

        p.prompt = prompt_styles.apply_styles_to_prompt(p.prompt, p.styles)
        p.negative_prompt = prompt_styles.apply_negative_styles_to_prompt(p.negative_prompt, p.styles)
        p.styles.clear()

        processing.fix_seed(p)

        project = Project(Path(global_options.output.output_dir) / inputs["project_name"])
        project.load()

        if not inputs["continue_from_last_frame"]:
            project.delete_all_frames()
            project.delete_session_data()

        session = self._ui_to_session(inputs, p)

        if inputs["load_parameters"] and project.session_path.exists():
            session.load(project.session_path)

        if not self._verify_image_existence(p, session.initial_noise, session.pipeline.parallel):
            opts.data.update(opts_backup)

            return Processed(p, p.init_images)

        if not session.iteration.images:
            session.iteration.images[:] = [pil_to_np(x) for x in p.init_images]

        state.job_count = inputs["iter_count"]

        for i in range(inputs["iter_count"]):
            logging.info(f"Iteration {i + 1} / {inputs['iter_count']}")

            start_time = perf_counter()

            state.job = "Temporal main loop"
            state.job_no = i

            if not session.pipeline.run(session):
                break

            if i % global_options.output.autosave_every_n_iterations == 0:
                session.save(project.session_path)
                project.save()

            end_time = perf_counter()

            logging.info(f"Iteration took {end_time - start_time:.6f} second(s)")

        session.pipeline.finalize(session)
        session.save(project.session_path)
        project.save()

        state.end()

        opts.data.update(opts_backup)

        return Processed(p, [np_to_pil(x) for x in session.iteration.images])

    @staticmethod
    def _verify_image_existence(p: StableDiffusionProcessingImg2Img, initial_noise: InitialNoiseParams, count: int) -> bool:
        if not p.init_images or not isinstance(p.init_images[0], PILImage):
            noises = [generate_value_noise(
                (p.height, p.width, 3),
                initial_noise.scale,
                initial_noise.octaves,
                initial_noise.lacunarity,
                initial_noise.persistence,
                p.seed + i,
            ) for i in range(count)]

            if initial_noise.factor < 1.0:
                if not (processed_images := process_images(
                    copy_with_overrides(p,
                        denoising_strength = 1.0 - initial_noise.factor,
                        do_not_save_samples = True,
                        do_not_save_grid = True,
                    ),
                    [(np_to_pil(x), p.seed + i, 1) for i, x in enumerate(noises)],
                    global_options.processing.pixels_per_batch,
                    True,
                )):
                    return False

                p.init_images = [image_array[0] for image_array in processed_images]

            else:
                p.init_images = noises

        elif len(p.init_images) != count:
            p.init_images = [p.init_images[0]] * count

        if opts.img2img_color_correction:
            p.color_corrections = [processing.setup_color_correction(x) for x in p.init_images]

        return True
