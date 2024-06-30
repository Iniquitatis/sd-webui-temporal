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
from temporal.project import render_project_video
from temporal.session import InitialNoiseParams
from temporal.shared import shared
from temporal.ui import CallbackInputs, CallbackOutputs, UI
from temporal.ui.fs_store_list import FSStoreList
from temporal.ui.gradio_widget import GradioWidget
from temporal.ui.options_editor import OptionsEditor
from temporal.ui.project_editor import ProjectEditor
from temporal.ui.video_renderer_editor import VideoRendererEditor
from temporal.utils import logging
from temporal.utils.fs import load_text
from temporal.utils.image import PILImage, ensure_image_dims, np_to_pil, pil_to_np
from temporal.utils.numpy import generate_value_noise
from temporal.utils.object import copy_with_overrides
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

        ui.add("preset", FSStoreList(label = "Preset", store = shared.preset_store, features = ["load", "save", "rename", "delete"]))

        @ui.callback("preset", "load", ["preset", "group:preset"], ["group:preset"])
        def _(inputs: CallbackInputs) -> CallbackOutputs:
            inputs |= inputs.pop("preset").data

            return {k: {"value": v} for k, v in inputs.items()}

        @ui.callback("preset", "save", ["group:preset"], ["preset"])
        def _(inputs: CallbackInputs) -> CallbackOutputs:
            return {"preset": {"value": Preset(inputs)}}

        ui.add("project", ProjectEditor(store = shared.project_store), groups = ["preset", "project"])

        with ui.add("", GradioWidget(gr.Tab, label = "Session")):
            ui.add("load_parameters", GradioWidget(gr.Checkbox, label = "Load parameters", value = True), groups = ["preset", "project"])
            ui.add("continue_from_last_frame", GradioWidget(gr.Checkbox, label = "Continue from last frame", value = True), groups = ["preset", "project"])
            ui.add("iter_count", GradioWidget(gr.Number, label = "Iteration count", precision = 0, minimum = 1, step = 1, value = 100), groups = ["preset", "project"])

        with ui.add("", GradioWidget(gr.Tab, label = "Video Rendering")):
            ui.add("video_renderer", VideoRendererEditor(value = shared.video_renderer), groups = ["preset"])
            ui.add("video_parallel_index", GradioWidget(gr.Number, label = "Parallel index", precision = 0, minimum = 1, step = 1, value = 1), groups = ["preset"])

            with ui.add("", GradioWidget(gr.Row)):
                ui.add("render_draft", GradioWidget(gr.Button, value = "Render draft"))
                ui.add("render_final", GradioWidget(gr.Button, value = "Render final"))

                def render_video(inputs: CallbackInputs, is_final: bool) -> Iterator[CallbackOutputs]:
                    yield {
                        "render_draft": {"interactive": False},
                        "render_final": {"interactive": False},
                    }

                    shared.video_renderer = inputs["video_renderer"]

                    video_path = render_project_video(
                        inputs["project"].path,
                        shared.video_renderer,
                        is_final,
                        inputs["video_parallel_index"],
                    )
                    wait_until(lambda: not video_render_queue.busy)

                    yield {
                        "render_draft": {"interactive": True},
                        "render_final": {"interactive": True},
                        "video_preview": {"value": video_path.as_posix()},
                    }

                @ui.callback("render_draft", "click", ["project", "video_renderer", "video_parallel_index"], ["render_draft", "render_final", "video_preview"])
                def _(inputs: CallbackInputs) -> Iterator[CallbackOutputs]:
                    yield from render_video(inputs, False)

                @ui.callback("render_final", "click", ["project", "video_renderer", "video_parallel_index"], ["render_draft", "render_final", "video_preview"])
                def _(inputs: CallbackInputs) -> Iterator[CallbackOutputs]:
                    yield from render_video(inputs, True)

            ui.add("video_preview", GradioWidget(gr.Video, label = "Preview", format = "mp4", interactive = False))

        with ui.add("", GradioWidget(gr.Tab, label = "Measuring")):
            ui.add("measuring_parallel_index", GradioWidget(gr.Number, label = "Parallel index", precision = 0, minimum = 1, step = 1, value = 1), groups = ["preset"])
            ui.add("render_graphs", GradioWidget(gr.Button, value = "Render graphs"))
            ui.add("graph_gallery", GradioWidget(gr.Gallery, label = "Graphs", columns = 4, object_fit = "contain", preview = True))

            @ui.callback("render_graphs", "click", ["project", "measuring_parallel_index"], ["graph_gallery"])
            def _(inputs: CallbackInputs) -> CallbackOutputs:
                project = inputs["project"]

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

        return ui.finalize(["group:project"])

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

        project = inputs["project"]

        session = project.session
        session.options = opts
        session.processing = p
        session.controlnet_units = get_cn_units(p)

        if inputs["load_parameters"]:
            project.load(project.path)

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
