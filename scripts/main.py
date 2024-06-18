from time import perf_counter
from typing import Any, Callable

import gradio as gr

from modules import scripts, shared as webui_shared
from modules.options import Options
from modules.processing import Processed, StableDiffusionProcessingImg2Img, fix_seed
from modules.styles import StyleDatabase
from modules.shared_state import State
from modules.ui_components import ToolButton

from temporal.blend_modes import BLEND_MODES
from temporal.global_options import OptionCategory
from temporal.image_filters import ImageFilter
from temporal.interop import EXTENSION_DIR, get_cn_units
from temporal.pipeline import PIPELINE_MODULES
from temporal.project import Project, render_project_video
from temporal.session import InitialNoiseParams
from temporal.shared import shared
from temporal.ui import UI
from temporal.ui.module_list import ModuleAccordion, ModuleAccordionSpecialCheckbox, ModuleList
from temporal.utils import logging
from temporal.utils.collection import get_first_element, get_next_element
from temporal.utils.fs import load_text
from temporal.utils.image import PILImage, np_to_pil, pil_to_np
from temporal.utils.numpy import generate_value_noise
from temporal.utils.object import copy_with_overrides, get_property_by_path, set_property_by_path
from temporal.utils.time import wait_until
from temporal.video_filters import VIDEO_FILTERS
from temporal.video_renderer import video_render_queue
from temporal.web_ui import process_images


# FIXME: To shut up the type checker
opts: Options = getattr(webui_shared, "opts")
prompt_styles: StyleDatabase = getattr(webui_shared, "prompt_styles")
state: State = getattr(webui_shared, "state")


class TemporalScript(scripts.Script):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        shared.init(EXTENSION_DIR / "settings", EXTENSION_DIR / "presets")

    def title(self) -> str:
        return "Temporal"

    def show(self, is_img2img: bool) -> Any:
        return is_img2img

    def ui(self, is_img2img: bool) -> Any:
        self._ui = ui = UI(self.elem_id)

        with ui.elem("", gr.Row):
            ui.elem("preset_name", gr.Dropdown, label = "Preset", choices = shared.preset_store.preset_names, allow_custom_value = True, value = get_first_element(shared.preset_store.preset_names, ""))
            ui.elem("refresh_presets", ToolButton, value = "\U0001f504")
            ui.elem("load_preset", ToolButton, value = "\U0001f4c2")
            ui.elem("save_preset", ToolButton, value = "\U0001f4be")
            ui.elem("delete_preset", ToolButton, value = "\U0001f5d1\ufe0f")

            @ui.callback("refresh_presets", "click", [], ["preset_name"])
            def _(_):
                shared.preset_store.refresh_presets()

                return {"preset_name": gr.update(choices = shared.preset_store.preset_names)}

            @ui.callback("load_preset", "click", ["preset_name", "group:preset"], ["group:preset"])
            def _(inputs):
                if inputs["preset_name"] not in shared.preset_store.preset_names:
                    return {}

                preset_name = inputs.pop("preset_name")

                inputs |= shared.preset_store.open_preset(preset_name).data

                return {k: gr.update(value = v) for k, v in inputs.items()}

            @ui.callback("save_preset", "click", ["preset_name", "group:preset"], ["preset_name"])
            def _(inputs):
                preset_name = inputs.pop("preset_name")

                shared.preset_store.save_preset(preset_name, inputs)

                return {"preset_name": gr.update(choices = shared.preset_store.preset_names, value = preset_name)}

            @ui.callback("delete_preset", "click", ["preset_name"], ["preset_name"])
            def _(inputs):
                if inputs["preset_name"] not in shared.preset_store.preset_names:
                    return {}

                preset_name = inputs["preset_name"]
                new_name = get_next_element(shared.preset_store.preset_names, preset_name, "untitled")

                shared.preset_store.delete_preset(preset_name)

                return {"preset_name": gr.update(choices = shared.preset_store.preset_names, value = new_name)}

        with ui.elem("", gr.Tab, label = "Project"):
            with ui.elem("", gr.Row):
                ui.elem("project_name", gr.Dropdown, label = "Project", choices = shared.project_store.project_names, allow_custom_value = True, value = get_first_element(shared.project_store.project_names, "untitled"), groups = ["preset", "project"])
                ui.elem("refresh_projects", ToolButton, value = "\U0001f504")
                ui.elem("load_project", ToolButton, value = "\U0001f4c2")
                ui.elem("delete_project", ToolButton, value = "\U0001f5d1\ufe0f")

                # FIXME: `change` makes typing slower, but `select` won't work until user clicks an appropriate item
                @ui.callback("project_name", "change", ["project_name"], ["project_description", "project_gallery", "project_gallery_parallel_index"])
                def _(inputs):
                    if inputs["project_name"] not in shared.project_store.project_names:
                        return {}

                    project = shared.project_store.open_project(inputs["project_name"])

                    return {
                        "project_description": gr.update(value = project.get_description()),
                        "project_gallery": gr.update(value = project.list_all_frame_paths()[:shared.options.ui.gallery_size]),
                        "project_gallery_page_index": gr.update(value = 1),
                        "project_gallery_parallel_index": gr.update(value = 1),
                    }

                @ui.callback("refresh_projects", "click", [], ["project_name"])
                def _(_):
                    shared.project_store.refresh_projects()

                    return {"project_name": gr.update(choices = shared.project_store.project_names)}

                @ui.callback("load_project", "click", ["project_name"], ["group:session"])
                def _(inputs):
                    if inputs["project_name"] not in shared.project_store.project_names:
                        return {}

                    project = shared.project_store.open_project(inputs["project_name"])

                    return {id: gr.update(value = get_property_by_path(project.session, id)) for id in self._ui.parse_ids(["group:session"])}

                @ui.callback("delete_project", "click", ["project_name"], ["project_name"])
                def _(inputs):
                    if inputs["project_name"] not in shared.project_store.project_names:
                        return {}

                    project_name = inputs["project_name"]
                    new_name = get_next_element(shared.project_store.project_names, project_name, "untitled")

                    shared.project_store.delete_project(project_name)

                    return {"project_name": gr.update(choices = shared.project_store.project_names, value = new_name)}

            with ui.elem("", gr.Tab, label = "Session"):
                ui.elem("load_parameters", gr.Checkbox, label = "Load parameters", value = True, groups = ["preset", "project"])
                ui.elem("continue_from_last_frame", gr.Checkbox, label = "Continue from last frame", value = True, groups = ["preset", "project"])
                ui.elem("iter_count", gr.Number, label = "Iteration count", precision = 0, minimum = 1, step = 1, value = 100, groups = ["preset", "project"])

            with ui.elem("", gr.Tab, label = "Information"):
                ui.elem("project_description", gr.Textbox, label = "Description", lines = 5, max_lines = 5, interactive = False)
                ui.elem("project_gallery", gr.Gallery, label = "Gallery", columns = 4, object_fit = "contain", preview = True)

                with gr.Row():
                    ui.elem("project_gallery_page_previous", ToolButton, value = "<")
                    ui.elem("project_gallery_page_index", gr.Number, label = "Page", precision = 0, minimum = 1, step = 1, value = 1)
                    ui.elem("project_gallery_page_next", ToolButton, value = ">")

                with gr.Row():
                    ui.elem("project_gallery_parallel_previous", ToolButton, value = "<")
                    ui.elem("project_gallery_parallel_index", gr.Number, label = "Parallel", precision = 0, minimum = 1, step = 1, value = 1)
                    ui.elem("project_gallery_parallel_next", ToolButton, value = ">")

                def navigate_gallery(inputs: dict[str, Any], page_offset: int, parallel_offset: int) -> dict[str, Any]:
                    if inputs["project_name"] not in shared.project_store.project_names:
                        return {}

                    project = shared.project_store.open_project(inputs["project_name"])
                    page = max(inputs["project_gallery_page_index"] + page_offset, 1)
                    parallel = max(inputs["project_gallery_parallel_index"] + parallel_offset, 1)
                    gallery_size = shared.options.ui.gallery_size

                    return {
                        "project_gallery": gr.update(value = project.list_all_frame_paths(parallel)[(page - 1) * gallery_size:page * gallery_size]),
                        "project_gallery_page_index": gr.update(value = page),
                        "project_gallery_parallel_index": gr.update(value = parallel),
                    }

                inputs = [
                    "project_name",
                    "project_gallery_page_index",
                    "project_gallery_parallel_index",
                ]
                outputs = [
                    "project_gallery",
                    "project_gallery_page_index",
                    "project_gallery_parallel_index",
                ]

                @ui.callback("project_gallery_page_previous", "click", inputs, outputs)
                def _(inputs):
                    return navigate_gallery(inputs, -1, 0)

                @ui.callback("project_gallery_page_index", "change", inputs, outputs)
                def _(inputs):
                    return navigate_gallery(inputs, 0, 0)

                @ui.callback("project_gallery_page_next", "click", inputs, outputs)
                def _(inputs):
                    return navigate_gallery(inputs, 1, 0)

                @ui.callback("project_gallery_parallel_previous", "click", inputs, outputs)
                def _(inputs):
                    return navigate_gallery(inputs, 0, -1)

                @ui.callback("project_gallery_parallel_index", "change", inputs, outputs)
                def _(inputs):
                    return navigate_gallery(inputs, 0, 0)

                @ui.callback("project_gallery_parallel_next", "click", inputs, outputs)
                def _(inputs):
                    return navigate_gallery(inputs, 0, 1)

            with ui.elem("", gr.Tab, label = "Tools"):
                with ui.elem("", gr.Row):
                    ui.elem("new_project_name", gr.Textbox, label = "New name", value = "")
                    ui.elem("confirm_project_rename", ToolButton, value = "\U00002714\ufe0f")

                    @ui.callback("confirm_project_rename", "click", ["project_name", "new_project_name"], ["project_name"])
                    def _(inputs):
                        if inputs["project_name"] not in shared.project_store.project_names:
                            return {}

                        shared.project_store.rename_project(inputs["project_name"], inputs["new_project_name"])

                        return {"project_name": gr.update(choices = shared.project_store.project_names, value = inputs["new_project_name"])}

                ui.elem("delete_intermediate_frames", gr.Button, value = "Delete intermediate frames")
                ui.elem("delete_session_data", gr.Button, value = "Delete session data")

                @ui.callback("delete_intermediate_frames", "click", ["project_name"], ["project_gallery"])
                def _(inputs):
                    if inputs["project_name"] not in shared.project_store.project_names:
                        return {}

                    project = shared.project_store.open_project(inputs["project_name"])
                    project.delete_intermediate_frames()

                    return {
                        "project_description": gr.update(value = project.get_description()),
                        "project_gallery": gr.update(value = project.list_all_frame_paths()[:shared.options.ui.gallery_size]),
                    }

                @ui.callback("delete_session_data", "click", ["project_name"], [])
                def _(inputs):
                    if inputs["project_name"] not in shared.project_store.project_names:
                        return {}

                    shared.project_store.open_project(inputs["project_name"]).delete_session_data()

                    return {}

        with ui.elem("", gr.Tab, label = "Pipeline"):
            with ui.elem("", gr.Accordion, label = "Initial noise", open = False):
                ui.elem("initial_noise.factor", gr.Slider, label = "Factor", minimum = 0.0, maximum = 1.0, step = 0.01, value = 0.0, groups = ["preset", "session"])
                ui.elem("initial_noise.scale", gr.Slider, label = "Scale", precision = 0, minimum = 1, maximum = 1024, step = 1, value = 1, groups = ["preset", "session"])
                ui.elem("initial_noise.octaves", gr.Slider, label = "Octaves", precision = 0, minimum = 1, maximum = 10, step = 1, value = 1, groups = ["preset", "session"])
                ui.elem("initial_noise.lacunarity", gr.Slider, label = "Lacunarity", minimum = 0.01, maximum = 4.0, step = 0.01, value = 2.0, groups = ["preset", "session"])
                ui.elem("initial_noise.persistence", gr.Slider, label = "Persistence", minimum = 0.0, maximum = 1.0, step = 0.01, value = 0.5, groups = ["preset", "session"])

            sorted_modules = dict(sorted(PIPELINE_MODULES.items(), key = lambda x: f"{x[1].icon} {x[1].id}"))

            ui.elem("pipeline.parallel", gr.Number, label = "Parallel", precision = 0, minimum = 1, step = 1, value = 1, groups = ["preset", "session"])

            with ui.elem("pipeline.module_order", ModuleList, keys = sorted_modules.keys(), groups = ["preset", "session"]):
                for id, module in sorted_modules.items():
                    with ui.elem(f"pipeline.modules['{id}'].enabled", ModuleAccordion, label = f"{module.icon} {module.name}", key = id, value = False, open = False, groups = ["preset", "session"]):
                        elem_id = f"preview_{id}"

                        def make_value_getter(id: str) -> Callable[[], bool]:
                            return lambda: shared.previewed_modules[id]

                        ui.elem(elem_id, ModuleAccordionSpecialCheckbox, value = make_value_getter(id), elem_classes = ["temporal-visibility-checkbox"], groups = ["preset"])

                        def make_callback(id: str, elem_id: str) -> None:
                            @ui.callback(elem_id, "change", [elem_id], [])
                            def _(inputs):
                                shared.previewed_modules[id] = inputs[elem_id]
                                return {}

                        make_callback(id, elem_id)

                        if issubclass(module, ImageFilter):
                            with ui.elem("", gr.Row):
                                ui.elem(f"pipeline.modules['{id}'].amount", gr.Slider, label = "Amount", minimum = 0.0, maximum = 1.0, step = 0.01, value = 1.0, groups = ["preset", "session"])
                                ui.elem(f"pipeline.modules['{id}'].amount_relative", gr.Checkbox, label = "Relative", value = False, groups = ["preset", "session"])

                            ui.elem(f"pipeline.modules['{id}'].blend_mode", gr.Dropdown, label = "Blend mode", choices = list(BLEND_MODES.keys()), value = get_first_element(BLEND_MODES), groups = ["preset", "session"])

                            with ui.elem("", gr.Tab, label = "Parameters"):
                                if module.__ui_params__:
                                    for param in module.__ui_params__.values():
                                        ui.elem(f"pipeline.modules['{id}'].{param.key}", param.gr_type, label = param.name, **param.kwargs, groups = ["preset", "session"])
                                else:
                                    ui.elem("", gr.Markdown, value = "_This filter has no available parameters._")

                            with ui.elem("", gr.Tab, label = "Mask"):
                                ui.elem(f"pipeline.modules['{id}'].mask.image", gr.Image, label = "Image", type = "numpy", image_mode = "L", interactive = True, groups = ["preset", "session"])
                                ui.elem(f"pipeline.modules['{id}'].mask.normalized", gr.Checkbox, label = "Normalized", value = False, groups = ["preset", "session"])
                                ui.elem(f"pipeline.modules['{id}'].mask.inverted", gr.Checkbox, label = "Inverted", value = False, groups = ["preset", "session"])
                                ui.elem(f"pipeline.modules['{id}'].mask.blurring", gr.Slider, label = "Blurring", minimum = 0.0, maximum = 50.0, step = 0.1, value = 0.0, groups = ["preset", "session"])

                        else:
                            for param in module.__ui_params__.values():
                                ui.elem(f"pipeline.modules['{id}'].{param.key}", param.gr_type, label = param.name, **param.kwargs, groups = ["preset", "session"])

        with ui.elem("", gr.Tab, label = "Video Rendering"):
            ui.elem("video_renderer.fps", gr.Slider, label = "Frames per second", precision = 0, minimum = 1, maximum = 60, step = 1, value = 30, groups = ["preset", "video"])
            ui.elem("video_renderer.looping", gr.Checkbox, label = "Looping", value = False, groups = ["preset", "video"])

            with ui.elem("video_renderer.filter_order", ModuleList, keys = VIDEO_FILTERS.keys(), groups = ["preset", "video"]):
                for id, filter in VIDEO_FILTERS.items():
                    with ui.elem(f"video_renderer.filters['{id}'].enabled", ModuleAccordion, label = filter.name, key = id, value = False, open = False, groups = ["preset", "video"]):
                        for param in filter.__ui_params__.values():
                            ui.elem(f"video_renderer.filters['{id}'].{param.key}", param.gr_type, label = param.name, **param.kwargs, groups = ["preset", "video"])

            ui.elem("video_parallel_index", gr.Number, label = "Parallel index", precision = 0, minimum = 1, step = 1, value = 1, groups = ["preset"])

            with ui.elem("", gr.Row):
                ui.elem("render_draft", gr.Button, value = "Render draft")
                ui.elem("render_final", gr.Button, value = "Render final")

                def render_video(inputs, is_final):
                    if inputs["project_name"] not in shared.project_store.project_names:
                        return {}

                    yield {
                        "render_draft": gr.update(interactive = False),
                        "render_final": gr.update(interactive = False),
                    }

                    project_name = inputs.pop("project_name")
                    parallel_index = inputs.pop("video_parallel_index")

                    for key, value in inputs.items():
                        set_property_by_path(shared, key, value)

                    video_path = render_project_video(
                        shared.options.output.output_dir / project_name,
                        shared.video_renderer,
                        is_final,
                        parallel_index,
                    )
                    wait_until(lambda: not video_render_queue.busy)

                    yield {
                        "render_draft": gr.update(interactive = True),
                        "render_final": gr.update(interactive = True),
                        "video_preview": video_path.as_posix(),
                    }

                @ui.callback("render_draft", "click", ["project_name", "video_parallel_index", "group:video"], ["render_draft", "render_final", "video_preview"])
                def _(inputs):
                    yield from render_video(inputs, False)

                @ui.callback("render_final", "click", ["project_name", "video_parallel_index", "group:video"], ["render_draft", "render_final", "video_preview"])
                def _(inputs):
                    yield from render_video(inputs, True)

            ui.elem("video_preview", gr.Video, label = "Preview", format = "mp4", interactive = False)

        with ui.elem("", gr.Tab, label = "Measuring"):
            ui.elem("render_plots", gr.Button, value = "Render plots")
            ui.elem("metrics_plots", gr.Gallery, label = "Plots", columns = 4, object_fit = "contain", preview = True)

            @ui.callback("render_plots", "click", ["project_name"], ["metrics_plots"])
            def _(inputs):
                if inputs["project_name"] not in shared.project_store.project_names:
                    return {}

                project = shared.project_store.open_project(inputs["project_name"])

                return {"metrics_plots": gr.update(value = list(project.session.pipeline.modules["measuring"].metrics.plot().values()))}

        with ui.elem("", gr.Tab, label = "Settings"):
            ui.elem("apply_settings", gr.Button, value = "Apply")

            @ui.callback("apply_settings", "click", ["group:options"], [])
            def _(inputs):
                for key, ui_value in inputs.items():
                    set_property_by_path(shared.options, key, ui_value)

                shared.options.save(EXTENSION_DIR / "settings")

                return {}

            for field in shared.options.__fields__.values():
                if not isinstance(category := getattr(shared.options, field.key), OptionCategory):
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

        return ui.finalize(["group:project", "group:session"])

    def run(self, p: StableDiffusionProcessingImg2Img, *args: Any) -> Any:
        return self._process(p, {name: arg for name, arg in zip(self._ui.parse_ids(["group:project", "group:session"]), args)})

    def _process(self, p: StableDiffusionProcessingImg2Img, inputs: dict[str, Any]) -> Processed:
        opts_backup = opts.data.copy()

        opts.save_to_dirs = False

        if shared.options.live_preview.show_only_finished_images:
            opts.show_progress_every_n_steps = -1

        p.prompt = prompt_styles.apply_styles_to_prompt(p.prompt, p.styles)
        p.negative_prompt = prompt_styles.apply_negative_styles_to_prompt(p.negative_prompt, p.styles)
        p.styles.clear()

        fix_seed(p)

        project = Project(shared.options.output.output_dir / inputs["project_name"], inputs["project_name"])

        session = project.session
        session.options = opts
        session.processing = p
        session.controlnet_units = get_cn_units(p)
        session.project = project

        if inputs["load_parameters"]:
            project.load(project.path)
        else:
            for id, value in inputs.items():
                if self._ui.is_in_group(id, "session"):
                    set_property_by_path(session, id, value)

        if not inputs["continue_from_last_frame"]:
            project.delete_all_frames()
            project.delete_session_data()

        if not self._verify_image_existence(p, session.initial_noise, session.pipeline.parallel):
            opts.data.update(opts_backup)

            return Processed(p, p.init_images)

        if not session.iteration.images:
            session.iteration.images[:] = [pil_to_np(x) for x in p.init_images]

        last_images = session.iteration.images.copy()

        state.job_count = inputs["iter_count"]

        for i in range(inputs["iter_count"]):
            logging.info(f"Iteration {i + 1} / {inputs['iter_count']}")

            start_time = perf_counter()

            state.job = "Temporal main loop"
            state.job_no = i

            if not session.pipeline.run(session):
                break

            last_images = session.iteration.images.copy()

            if i % shared.options.output.autosave_every_n_iterations == 0:
                project.save(project.path)

            end_time = perf_counter()

            logging.info(f"Iteration took {end_time - start_time:.6f} second(s)")

        session.pipeline.finalize(session)
        project.save(project.path)

        state.end()

        opts.data.update(opts_backup)

        return Processed(p, [np_to_pil(x) for x in last_images])

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
                    shared.options.processing.pixels_per_batch,
                    True,
                )):
                    return False

                p.init_images = [image_array[0] for image_array in processed_images]

            else:
                p.init_images = noises

        elif len(p.init_images) != count:
            p.init_images = [p.init_images[0]] * count

        return True
