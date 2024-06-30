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
        self._ui = UI()

        preset = FSStoreList(label = "Preset", store = shared.preset_store, features = ["load", "save", "rename", "delete"])
        project = ProjectEditor(store = shared.project_store)

        with GradioWidget(gr.Tab, label = "Session"):
            load_parameters = GradioWidget(gr.Checkbox, label = "Load parameters", value = True)
            continue_from_last_frame = GradioWidget(gr.Checkbox, label = "Continue from last frame", value = True)
            iter_count = GradioWidget(gr.Number, label = "Iteration count", precision = 0, minimum = 1, step = 1, value = 100)

        with GradioWidget(gr.Tab, label = "Video Rendering"):
            video_renderer = VideoRendererEditor(value = shared.video_renderer)
            video_parallel_index = GradioWidget(gr.Number, label = "Parallel index", precision = 0, minimum = 1, step = 1, value = 1)

            with GradioWidget(gr.Row):
                render_draft = GradioWidget(gr.Button, value = "Render draft")
                render_final = GradioWidget(gr.Button, value = "Render final")

            video_preview = GradioWidget(gr.Video, label = "Preview", format = "mp4", interactive = False)

        with GradioWidget(gr.Tab, label = "Measuring"):
            measuring_parallel_index = GradioWidget(gr.Number, label = "Parallel index", precision = 0, minimum = 1, step = 1, value = 1)
            render_graphs = GradioWidget(gr.Button, value = "Render graphs")
            graph_gallery = GradioWidget(gr.Gallery, label = "Graphs", columns = 4, object_fit = "contain", preview = True)

        with GradioWidget(gr.Tab, label = "Settings"):
            apply_settings = GradioWidget(gr.Button, value = "Apply")
            options = OptionsEditor(value = shared.options)

        with GradioWidget(gr.Tab, label = "Help"):
            for file_name, title in [
                ("main.md", "Main"),
                ("tab_project.md", "Project tab"),
                ("tab_pipeline.md", "Pipeline tab"),
                ("tab_video_rendering.md", "Video Rendering tab"),
                ("tab_measuring.md", "Measuring tab"),
                ("tab_settings.md", "Settings tab"),
            ]:
                with GradioWidget(gr.Accordion, label = title, open = False):
                    GradioWidget(gr.Markdown, value = load_text(EXTENSION_DIR / "docs" / "temporal" / file_name, ""))

        @preset.callback("load", [preset, project, load_parameters, continue_from_last_frame, iter_count, video_renderer], [project, load_parameters, continue_from_last_frame, iter_count, video_renderer])
        def _(inputs: CallbackInputs) -> CallbackOutputs:
            data = inputs[preset].data

            return {
                project: {"value": data["project"]},
                load_parameters: {"value": data["load_parameters"]},
                continue_from_last_frame: {"value": data["continue_from_last_frame"]},
                iter_count: {"value": data["iter_count"]},
                video_renderer: {"value": data["video_renderer"]},
            }

        @preset.callback("save", [project, load_parameters, continue_from_last_frame, iter_count, video_renderer], [preset])
        def _(inputs: CallbackInputs) -> CallbackOutputs:
            return {preset: {"value": Preset({
                "project": inputs[project],
                "load_parameters": inputs[load_parameters],
                "continue_from_last_frame": inputs[continue_from_last_frame],
                "iter_count": inputs[iter_count],
                "video_renderer": inputs[video_renderer],
            })}}

        def render_video(inputs: CallbackInputs, is_final: bool) -> Iterator[CallbackOutputs]:
            yield {
                render_draft: {"interactive": False},
                render_final: {"interactive": False},
            }

            shared.video_renderer = inputs[video_renderer]

            video_path = render_project_video(
                inputs[project].path,
                shared.video_renderer,
                is_final,
                inputs[video_parallel_index],
            )
            wait_until(lambda: not video_render_queue.busy)

            yield {
                render_draft: {"interactive": True},
                render_final: {"interactive": True},
                video_preview: {"value": video_path.as_posix()},
            }

        @render_draft.callback("click", [project, video_renderer, video_parallel_index], [render_draft, render_final, video_preview])
        def _(inputs: CallbackInputs) -> Iterator[CallbackOutputs]:
            yield from render_video(inputs, False)

        @render_final.callback("click", [project, video_renderer, video_parallel_index], [render_draft, render_final, video_preview])
        def _(inputs: CallbackInputs) -> Iterator[CallbackOutputs]:
            yield from render_video(inputs, True)

        @render_graphs.callback("click", [project, measuring_parallel_index], [graph_gallery])
        def _(inputs: CallbackInputs) -> CallbackOutputs:
            return {graph_gallery: {"value": [
                x.plot(inputs[measuring_parallel_index] - 1)
                for x in inputs[project].session.pipeline.modules.values()
                if isinstance(x, MeasuringModule) and x.enabled
            ]}}

        @apply_settings.callback("click", [options], [])
        def _(inputs: CallbackInputs) -> CallbackOutputs:
            shared.options = inputs[options]
            shared.options.save(EXTENSION_DIR / "settings")

            return {}

        return self._ui.finalize(project, load_parameters, continue_from_last_frame, iter_count)

    def run(self, p: StableDiffusionProcessingImg2Img, *args: Any) -> Any:
        project: Project
        load_parameters: bool
        continue_from_last_frame: bool
        iter_count: int

        project, load_parameters, continue_from_last_frame, iter_count = self._ui.recombine(*args)

        opts_backup = opts.data.copy()

        opts.save_to_dirs = False

        if shared.options.live_preview.show_only_finished_images:
            opts.show_progress_every_n_steps = -1

        p.prompt = prompt_styles.apply_styles_to_prompt(p.prompt, p.styles)
        p.negative_prompt = prompt_styles.apply_negative_styles_to_prompt(p.negative_prompt, p.styles)
        p.styles.clear()

        fix_seed(p)

        session = project.session
        session.options = opts
        session.processing = p
        session.controlnet_units = get_cn_units(p)

        if load_parameters:
            project.load(project.path)

        if not continue_from_last_frame:
            project.delete_all_frames()
            project.delete_session_data()

        if not p.init_images or not isinstance(p.init_images[0], PILImage):
            noises = [generate_value_noise(
                (p.height, p.width, 3),
                session.initial_noise.noise.scale,
                session.initial_noise.noise.octaves,
                session.initial_noise.noise.lacunarity,
                session.initial_noise.noise.persistence,
                (p.seed if session.initial_noise.use_initial_seed else session.initial_noise.noise.seed) + i,
            ) for i in range(session.pipeline.parallel)]

            if session.initial_noise.factor < 1.0:
                if not (processed_images := process_images(
                    copy_with_overrides(p,
                        denoising_strength = 1.0 - session.initial_noise.factor,
                        do_not_save_samples = True,
                        do_not_save_grid = True,
                    ),
                    [(np_to_pil(x), p.seed + i, 1) for i, x in enumerate(noises)],
                    shared.options.processing.pixels_per_batch,
                    True,
                )):
                    opts.data.update(opts_backup)

                    return Processed(p, p.init_images)

                p.init_images = [image_array[0] for image_array in processed_images]

            else:
                p.init_images = noises

        elif len(p.init_images) != session.pipeline.parallel:
            p.init_images = [p.init_images[0]] * session.pipeline.parallel

        if not session.iteration.images:
            session.iteration.images[:] = [pil_to_np(ensure_image_dims(x, "RGB", (p.width, p.height))) for x in p.init_images]

        last_images = session.iteration.images.copy()

        state.job_count = iter_count

        for i in range(iter_count):
            logging.info(f"Iteration {i + 1} / {iter_count}")

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
