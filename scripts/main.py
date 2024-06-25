from time import perf_counter
from typing import Any, Callable, Iterator

import gradio as gr

from modules import scripts, shared as webui_shared
from modules.options import Options
from modules.processing import Processed, StableDiffusionProcessingImg2Img, fix_seed
from modules.shared_state import State
from modules.styles import StyleDatabase

from temporal.blend_modes import BLEND_MODES
from temporal.global_options import OptionCategory
from temporal.image_filters import ImageFilter
from temporal.interop import EXTENSION_DIR, get_cn_units
from temporal.pipeline import PIPELINE_MODULES
from temporal.preset import Preset
from temporal.project import Project, render_project_video
from temporal.session import InitialNoiseParams
from temporal.shared import shared
from temporal.ui import CallbackInputs, CallbackOutputs, UI
from temporal.ui.configurable_param_editor import ConfigurableParamEditor
from temporal.ui.dropdown import Dropdown
from temporal.ui.fs_store_list import FSStoreList
from temporal.ui.gradio_widget import GradioWidget
from temporal.ui.image_mask_editor import ImageMaskEditor
from temporal.ui.module_list import ModuleAccordion, ModuleAccordionSpecialCheckbox, ModuleList
from temporal.ui.noise_editor import NoiseEditor
from temporal.ui.paginator import Paginator
from temporal.utils import logging
from temporal.utils.collection import get_first_element
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
        self._ui = ui = UI()

        with ui.add("", GradioWidget(gr.Row)):
            ui.add("preset_name", FSStoreList(label = "Preset", store = shared.preset_store, features = ["load", "save", "rename", "delete"]))

            @ui.callback("preset_name", "load", ["preset_name", "group:preset"], ["group:preset"])
            def _(inputs: CallbackInputs) -> CallbackOutputs:
                inputs |= inputs.pop("preset_name").data

                return {k: {"value": v} for k, v in inputs.items()}

            @ui.callback("preset_name", "save", ["group:preset"], ["preset_name"])
            def _(inputs: CallbackInputs) -> CallbackOutputs:
                return {"preset_name": {"value": Preset(inputs)}}

        with ui.add("", GradioWidget(gr.Tab, label = "Project")):
            with ui.add("", GradioWidget(gr.Row)):
                ui.add("project_name", FSStoreList(label = "Project", store = shared.project_store, features = ["load", "rename", "delete"]), groups = ["preset", "project"])

                # FIXME: `change` makes typing slower, but `select` won't work until user clicks an appropriate item
                @ui.callback("project_name", "change", ["project_name"], ["project_description", "project_gallery", "project_gallery_page", "project_gallery_parallel"])
                def _(inputs: CallbackInputs) -> CallbackOutputs:
                    if inputs["project_name"] not in shared.project_store.entry_names:
                        return {}

                    project = shared.project_store.load_entry(inputs["project_name"])

                    return {
                        "project_description": {"value": project.get_description()},
                        "project_gallery": {"value": project.list_all_frame_paths()[:shared.options.ui.gallery_size]},
                        "project_gallery_page": {"value": 1},
                        "project_gallery_parallel": {"value": 1},
                    }

                @ui.callback("project_name", "load", ["project_name"], ["group:session"])
                def _(inputs: CallbackInputs) -> CallbackOutputs:
                    project = inputs["project_name"]

                    return {
                        id: {"value": get_property_by_path(project.session, id)}
                        for id in self._ui.parse_id("group:session")
                    }

            with ui.add("", GradioWidget(gr.Tab, label = "Session")):
                ui.add("load_parameters", GradioWidget(gr.Checkbox, label = "Load parameters", value = True), groups = ["preset", "project"])
                ui.add("continue_from_last_frame", GradioWidget(gr.Checkbox, label = "Continue from last frame", value = True), groups = ["preset", "project"])
                ui.add("iter_count", GradioWidget(gr.Number, label = "Iteration count", precision = 0, minimum = 1, step = 1, value = 100), groups = ["preset", "project"])

            with ui.add("", GradioWidget(gr.Tab, label = "Information")):
                ui.add("project_description", GradioWidget(gr.Textbox, label = "Description", lines = 5, max_lines = 5, interactive = False))
                ui.add("project_gallery", GradioWidget(gr.Gallery, label = "Gallery", columns = 4, object_fit = "contain", preview = True))
                ui.add("project_gallery_page", Paginator(label = "Page", minimum = 1, value = 1))
                ui.add("project_gallery_parallel", Paginator(label = "Parallel", minimum = 1, value = 1))

                def update_gallery(inputs: CallbackInputs) -> CallbackOutputs:
                    if inputs["project_name"] not in shared.project_store.entry_names:
                        return {}

                    project = shared.project_store.load_entry(inputs["project_name"])
                    page = inputs["project_gallery_page"]
                    parallel = inputs["project_gallery_parallel"]
                    gallery_size = shared.options.ui.gallery_size

                    return {"project_gallery": {"value": project.list_all_frame_paths(parallel)[(page - 1) * gallery_size:page * gallery_size]}}

                @ui.callback("project_gallery_page", "change", ["project_name", "project_gallery_page", "project_gallery_parallel"], ["project_gallery"])
                def _(inputs: CallbackInputs) -> CallbackOutputs:
                    return update_gallery(inputs)

                @ui.callback("project_gallery_parallel", "change", ["project_name", "project_gallery_page", "project_gallery_parallel"], ["project_gallery"])
                def _(inputs: CallbackInputs) -> CallbackOutputs:
                    return update_gallery(inputs)

            with ui.add("", GradioWidget(gr.Tab, label = "Tools")):
                ui.add("delete_intermediate_frames", GradioWidget(gr.Button, value = "Delete intermediate frames"))
                ui.add("delete_session_data", GradioWidget(gr.Button, value = "Delete session data"))

                @ui.callback("delete_intermediate_frames", "click", ["project_name"], ["project_gallery"])
                def _(inputs: CallbackInputs) -> CallbackOutputs:
                    if inputs["project_name"] not in shared.project_store.entry_names:
                        return {}

                    project = shared.project_store.load_entry(inputs["project_name"])
                    project.delete_intermediate_frames()

                    return {
                        "project_description": {"value": project.get_description()},
                        "project_gallery": {"value": project.list_all_frame_paths()[:shared.options.ui.gallery_size]},
                    }

                @ui.callback("delete_session_data", "click", ["project_name"], [])
                def _(inputs: CallbackInputs) -> CallbackOutputs:
                    if inputs["project_name"] not in shared.project_store.entry_names:
                        return {}

                    project = shared.project_store.load_entry(inputs["project_name"])
                    project.delete_session_data()
                    project.save(project.path)

                    return {}

        with ui.add("", GradioWidget(gr.Tab, label = "Pipeline")):
            with ui.add("", GradioWidget(gr.Accordion, label = "Initial noise", open = False)):
                ui.add("initial_noise.factor", GradioWidget(gr.Slider, label = "Factor", minimum = 0.0, maximum = 1.0, step = 0.01, value = 0.0), groups = ["preset", "session"])
                ui.add("initial_noise.noise", NoiseEditor(), groups = ["preset", "session"])
                ui.add("initial_noise.use_initial_seed", GradioWidget(gr.Checkbox, label = "Use initial seed", value = False), groups = ["preset", "session"])

            sorted_modules = dict(sorted(PIPELINE_MODULES.items(), key = lambda x: f"{x[1].icon} {x[1].id}"))

            ui.add("pipeline.parallel", GradioWidget(gr.Number, label = "Parallel", precision = 0, minimum = 1, step = 1, value = 1), groups = ["preset", "session"])

            with ui.add("pipeline.module_order", ModuleList(keys = sorted_modules.keys()), groups = ["preset", "session"]):
                for id, module in sorted_modules.items():
                    with ui.add(f"pipeline.modules['{id}'].enabled", ModuleAccordion(label = f"{module.icon} {module.name}", key = id, value = False, open = False), groups = ["preset", "session"]):
                        elem_id = f"preview_{id}"

                        def make_value_getter(id: str) -> Callable[[], bool]:
                            return lambda: shared.previewed_modules[id]

                        ui.add(elem_id, ModuleAccordionSpecialCheckbox(value = make_value_getter(id), classes = ["temporal-visibility-checkbox"]), groups = ["preset"])

                        def make_callback(id: str, elem_id: str) -> None:
                            @ui.callback(elem_id, "change", [elem_id], [])
                            def _(inputs: CallbackInputs) -> CallbackOutputs:
                                shared.previewed_modules[id] = inputs[elem_id]
                                return {}

                        make_callback(id, elem_id)

                        if issubclass(module, ImageFilter):
                            with ui.add("", GradioWidget(gr.Row)):
                                ui.add(f"pipeline.modules['{id}'].amount", GradioWidget(gr.Slider, label = "Amount", minimum = 0.0, maximum = 1.0, step = 0.01, value = 1.0), groups = ["preset", "session"])
                                ui.add(f"pipeline.modules['{id}'].amount_relative", GradioWidget(gr.Checkbox, label = "Relative", value = False), groups = ["preset", "session"])

                            ui.add(f"pipeline.modules['{id}'].blend_mode", Dropdown(label = "Blend mode", choices = [(x.id, x.name) for x in BLEND_MODES.values()], value = get_first_element(BLEND_MODES)), groups = ["preset", "session"])

                            with ui.add("", GradioWidget(gr.Tab, label = "Parameters")):
                                if module.__params__:
                                    for param in module.__params__.values():
                                        ui.add(f"pipeline.modules['{id}'].{param.key}", ConfigurableParamEditor(param = param), groups = ["preset", "session"])
                                else:
                                    ui.add("", GradioWidget(gr.Markdown, value = "_This filter has no available parameters._"))

                            with ui.add("", GradioWidget(gr.Tab, label = "Mask")):
                                ui.add(f"pipeline.modules['{id}'].mask", ImageMaskEditor(), groups = ["preset", "session"])

                        else:
                            for param in module.__params__.values():
                                ui.add(f"pipeline.modules['{id}'].{param.key}", ConfigurableParamEditor(param = param), groups = ["preset", "session"])

        with ui.add("", GradioWidget(gr.Tab, label = "Video Rendering")):
            ui.add("video_renderer.fps", GradioWidget(gr.Slider, label = "Frames per second", minimum = 1, maximum = 60, step = 1, value = 30), groups = ["preset", "video"])
            ui.add("video_renderer.looping", GradioWidget(gr.Checkbox, label = "Looping", value = False), groups = ["preset", "video"])

            with ui.add("video_renderer.filter_order", ModuleList(keys = VIDEO_FILTERS.keys()), groups = ["preset", "video"]):
                for id, filter in VIDEO_FILTERS.items():
                    with ui.add(f"video_renderer.filters['{id}'].enabled", ModuleAccordion(label = filter.name, key = id, value = False, open = False), groups = ["preset", "video"]):
                        for param in filter.__params__.values():
                            ui.add(f"video_renderer.filters['{id}'].{param.key}", ConfigurableParamEditor(param = param), groups = ["preset", "video"])

            ui.add("video_parallel_index", GradioWidget(gr.Number, label = "Parallel index", precision = 0, minimum = 1, step = 1, value = 1), groups = ["preset"])

            with ui.add("", GradioWidget(gr.Row)):
                ui.add("render_draft", GradioWidget(gr.Button, value = "Render draft"))
                ui.add("render_final", GradioWidget(gr.Button, value = "Render final"))

                def render_video(inputs: CallbackInputs, is_final: bool) -> Iterator[CallbackOutputs]:
                    if inputs["project_name"] not in shared.project_store.entry_names:
                        return {}

                    yield {
                        "render_draft": {"interactive": False},
                        "render_final": {"interactive": False},
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
                        "render_draft": {"interactive": True},
                        "render_final": {"interactive": True},
                        "video_preview": {"value": video_path.as_posix()},
                    }

                @ui.callback("render_draft", "click", ["project_name", "video_parallel_index", "group:video"], ["render_draft", "render_final", "video_preview"])
                def _(inputs: CallbackInputs) -> Iterator[CallbackOutputs]:
                    yield from render_video(inputs, False)

                @ui.callback("render_final", "click", ["project_name", "video_parallel_index", "group:video"], ["render_draft", "render_final", "video_preview"])
                def _(inputs: CallbackInputs) -> Iterator[CallbackOutputs]:
                    yield from render_video(inputs, True)

            ui.add("video_preview", GradioWidget(gr.Video, label = "Preview", format = "mp4", interactive = False))

        with ui.add("", GradioWidget(gr.Tab, label = "Measuring")):
            ui.add("render_plots", GradioWidget(gr.Button, value = "Render plots"))
            ui.add("metrics_plots", GradioWidget(gr.Gallery, label = "Plots", columns = 4, object_fit = "contain", preview = True))

            @ui.callback("render_plots", "click", ["project_name"], ["metrics_plots"])
            def _(inputs: CallbackInputs) -> CallbackOutputs:
                if inputs["project_name"] not in shared.project_store.entry_names:
                    return {}

                project = shared.project_store.load_entry(inputs["project_name"])

                return {"metrics_plots": {"value": list(project.session.pipeline.modules["measuring"].metrics.plot().values())}}

        with ui.add("", GradioWidget(gr.Tab, label = "Settings")):
            ui.add("apply_settings", GradioWidget(gr.Button, value = "Apply"))

            @ui.callback("apply_settings", "click", ["group:options"], [])
            def _(inputs: CallbackInputs) -> CallbackOutputs:
                for key, ui_value in inputs.items():
                    set_property_by_path(shared.options, key, ui_value)

                shared.options.save(EXTENSION_DIR / "settings")

                return {}

            for field in shared.options.__fields__.values():
                if not isinstance(category := getattr(shared.options, field.key), OptionCategory):
                    continue

                with ui.add("", GradioWidget(gr.Accordion, label = category.name, open = False)):
                    def make_param_getter(category: OptionCategory, key: str) -> Callable[[], Any]:
                        return lambda: getattr(category, key)

                    for param in category.__params__.values():
                        ui.add(f"{field.key}.{param.key}", ConfigurableParamEditor(param = param, value = make_param_getter(category, param.key)), groups = ["options"])

        with ui.add("", GradioWidget(gr.Tab, label = "Help")):
            for file_name, title in [
                ("main.md", "Main"),
                ("tab_project.md", "Project tab"),
                ("tab_pipeline.md", "Pipeline tab"),
                ("tab_video_rendering.md", "Video Rendering tab"),
                ("tab_measuring.md", "Measuring tab"),
                ("tab_settings.md", "Settings tab"),
                ("additional_notes.md", "Additional notes"),
            ]:
                with ui.add("", GradioWidget(gr.Accordion, label = title, open = False)):
                    ui.add("", GradioWidget(gr.Markdown, value = load_text(EXTENSION_DIR / "docs" / "temporal" / file_name, "")))

        return ui.finalize(["group:project", "group:session"])

    def run(self, p: StableDiffusionProcessingImg2Img, *args: Any) -> Any:
        return self._process(p, self._ui.recombine(*args))

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
                initial_noise.noise.scale,
                initial_noise.noise.octaves,
                initial_noise.noise.lacunarity,
                initial_noise.noise.persistence,
                (p.seed if initial_noise.use_initial_seed else initial_noise.noise.seed) + i,
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
