from time import perf_counter
from typing import Any, Iterator

import gradio as gr

from modules import scripts, shared as webui_shared
from modules.options import Options
from modules.processing import Processed, StableDiffusionProcessingImg2Img, fix_seed
from modules.shared_state import State
from modules.styles import StyleDatabase

from temporal.interop import EXTENSION_DIR, get_cn_units
from temporal.measuring_modules import MeasuringModule
from temporal.preset import Preset
from temporal.project import Project, render_project_video
from temporal.session import InitialNoiseParams
from temporal.shared import shared
from temporal.ui import CallbackInputs, CallbackOutputs, UI
from temporal.ui.fs_store_list import FSStoreList
from temporal.ui.gradio_widget import GradioWidget
from temporal.ui.initial_noise_editor import InitialNoiseEditor
from temporal.ui.options_editor import OptionsEditor
from temporal.ui.paginator import Paginator
from temporal.ui.pipeline_editor import PipelineEditor
from temporal.ui.video_renderer_editor import VideoRendererEditor
from temporal.utils import logging
from temporal.utils.fs import load_text
from temporal.utils.image import PILImage, ensure_image_dims, np_to_pil, pil_to_np
from temporal.utils.numpy import generate_value_noise
from temporal.utils.object import copy_with_overrides, get_property_by_path, set_property_by_path
from temporal.utils.time import wait_until
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

                @ui.callback("delete_intermediate_frames", "click", ["project_name"], ["project_description", "project_gallery"])
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
            ui.add("initial_noise", InitialNoiseEditor(), groups = ["preset", "session"])
            ui.add("pipeline", PipelineEditor(), groups = ["preset", "session"])

        with ui.add("", GradioWidget(gr.Tab, label = "Video Rendering")):
            ui.add("video_renderer", VideoRendererEditor(value = shared.video_renderer), groups = ["preset"])
            ui.add("video_parallel_index", GradioWidget(gr.Number, label = "Parallel index", precision = 0, minimum = 1, step = 1, value = 1), groups = ["preset"])

            with ui.add("", GradioWidget(gr.Row)):
                ui.add("render_draft", GradioWidget(gr.Button, value = "Render draft"))
                ui.add("render_final", GradioWidget(gr.Button, value = "Render final"))

                def render_video(inputs: CallbackInputs, is_final: bool) -> Iterator[CallbackOutputs]:
                    if inputs["project_name"] not in shared.project_store.entry_names:
                        # FIXME: Early return locks up the UI for some reason
                        yield {}

                    else:
                        yield {
                            "render_draft": {"interactive": False},
                            "render_final": {"interactive": False},
                        }

                        project_name = inputs.pop("project_name")
                        parallel_index = inputs.pop("video_parallel_index")

                        shared.video_renderer = inputs["video_renderer"]

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

                @ui.callback("render_draft", "click", ["project_name", "video_renderer", "video_parallel_index"], ["render_draft", "render_final", "video_preview"])
                def _(inputs: CallbackInputs) -> Iterator[CallbackOutputs]:
                    yield from render_video(inputs, False)

                @ui.callback("render_final", "click", ["project_name", "video_renderer", "video_parallel_index"], ["render_draft", "render_final", "video_preview"])
                def _(inputs: CallbackInputs) -> Iterator[CallbackOutputs]:
                    yield from render_video(inputs, True)

            ui.add("video_preview", GradioWidget(gr.Video, label = "Preview", format = "mp4", interactive = False))

        with ui.add("", GradioWidget(gr.Tab, label = "Measuring")):
            ui.add("measuring_parallel_index", GradioWidget(gr.Number, label = "Parallel index", precision = 0, minimum = 1, step = 1, value = 1), groups = ["preset"])
            ui.add("render_graphs", GradioWidget(gr.Button, value = "Render graphs"))
            ui.add("graph_gallery", GradioWidget(gr.Gallery, label = "Graphs", columns = 4, object_fit = "contain", preview = True))

            @ui.callback("render_graphs", "click", ["project_name", "measuring_parallel_index"], ["graph_gallery"])
            def _(inputs: CallbackInputs) -> CallbackOutputs:
                if inputs["project_name"] not in shared.project_store.entry_names:
                    return {}

                project = shared.project_store.load_entry(inputs["project_name"])

                return {"graph_gallery": {"value": [
                    x.plot(inputs["measuring_parallel_index"] - 1)
                    for x in project.session.pipeline.modules.values()
                    if isinstance(x, MeasuringModule) and x.enabled
                ]}}

        with ui.add("", GradioWidget(gr.Tab, label = "Settings")):
            ui.add("apply_settings", GradioWidget(gr.Button, value = "Apply"))
            ui.add("options", OptionsEditor(value = shared.options))

            @ui.callback("apply_settings", "click", ["options"], [])
            def _(inputs: CallbackInputs) -> CallbackOutputs:
                shared.options = inputs["options"]
                shared.options.save(EXTENSION_DIR / "settings")

                return {}

        with ui.add("", GradioWidget(gr.Tab, label = "Help")):
            for file_name, title in [
                ("main.md", "Main"),
                ("tab_project.md", "Project tab"),
                ("tab_pipeline.md", "Pipeline tab"),
                ("tab_video_rendering.md", "Video Rendering tab"),
                ("tab_measuring.md", "Measuring tab"),
                ("tab_settings.md", "Settings tab"),
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
            session.iteration.images[:] = [pil_to_np(ensure_image_dims(x, "RGB", (p.width, p.height))) for x in p.init_images]

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
