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
from temporal.project import Project
from temporal.shared import shared
from temporal.ui import CallbackInputs, CallbackOutputs, UI
from temporal.ui.fs_store_list import FSStoreList, FSStoreListEntry
from temporal.ui.gradio_widget import GradioWidget
from temporal.ui.options_editor import OptionsEditor
from temporal.ui.paginator import Paginator
from temporal.ui.project_editor import ProjectEditor
from temporal.ui.video_renderer_editor import VideoRendererEditor
from temporal.utils import logging
from temporal.utils.fs import load_text
from temporal.utils.image import PILImage, ensure_image_dims, np_to_pil, pil_to_np
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

        stored_preset = FSStoreList(label = "Preset", store = shared.preset_store, features = ["load", "save", "rename", "delete"])
        stored_project = FSStoreList(label = "Project", store = shared.project_store, features = ["load", "rename", "delete"])

        with GradioWidget(gr.Tab, label = "General"):
            load_parameters = GradioWidget(gr.Checkbox, label = "Load parameters", value = True)
            continue_from_last_frame = GradioWidget(gr.Checkbox, label = "Continue from last frame", value = True)
            iter_count = GradioWidget(gr.Number, label = "Iteration count", precision = 0, minimum = 1, step = 1, value = 100)

        with GradioWidget(gr.Tab, label = "Information"):
            description = GradioWidget(gr.Textbox, label = "Description", lines = 5, max_lines = 5, interactive = False)
            gallery = GradioWidget(gr.Gallery, label = "Gallery", columns = 4, object_fit = "contain", preview = True)
            gallery_page = Paginator(label = "Page", minimum = 1, value = 1)
            gallery_parallel = Paginator(label = "Parallel", minimum = 1, value = 1)

        with GradioWidget(gr.Tab, label = "Pipeline"):
            project = ProjectEditor()

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

        with GradioWidget(gr.Tab, label = "Tools"):
            delete_intermediate_frames = GradioWidget(gr.Button, value = "Delete intermediate frames")
            delete_session_data = GradioWidget(gr.Button, value = "Delete session data")

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

        @stored_preset.callback("load", [stored_preset], [stored_project, load_parameters, continue_from_last_frame, iter_count, project, video_renderer])
        def _(inputs: CallbackInputs) -> CallbackOutputs:
            data = inputs[stored_preset].data.data

            return {
                stored_project: {"value": data["stored_project"]},
                load_parameters: {"value": data["load_parameters"]},
                continue_from_last_frame: {"value": data["continue_from_last_frame"]},
                iter_count: {"value": data["iter_count"]},
                project: {"value": data["project"], "preview_states": data["preview_states"]},
                video_renderer: {"value": data["video_renderer"]},
            }

        @stored_preset.callback("save", [stored_project, load_parameters, continue_from_last_frame, iter_count, project, video_renderer], [stored_preset])
        def _(inputs: CallbackInputs) -> CallbackOutputs:
            return {stored_preset: {"value": Preset({
                "stored_project": inputs[stored_project].name,
                "load_parameters": inputs[load_parameters],
                "continue_from_last_frame": inputs[continue_from_last_frame],
                "iter_count": inputs[iter_count],
                "project": inputs[project],
                "preview_states": shared.previewed_modules,
                "video_renderer": inputs[video_renderer],
            })}}

        @stored_project.callback("change", [stored_project], [description, gallery, gallery_page, gallery_parallel])
        def _(inputs: CallbackInputs) -> CallbackOutputs:
            project_obj = inputs[stored_project].data

            return {
                description: {"value": project_obj.get_description()},
                gallery: {"value": project_obj.list_all_frame_paths()[:shared.options.ui.gallery_size]},
                gallery_page: {"value": 1},
                gallery_parallel: {"value": 1},
            }

        @stored_project.callback("load", [stored_project], [project])
        def _(inputs: CallbackInputs) -> CallbackOutputs:
            return {project: {"value": inputs[stored_project].data}}

        @gallery_page.callback("change", [stored_project, gallery_page, gallery_parallel], [gallery])
        @gallery_parallel.callback("change", [stored_project, gallery_page, gallery_parallel], [gallery])
        def _(inputs: CallbackInputs) -> CallbackOutputs:
            project_obj = inputs[stored_project].data
            page = inputs[gallery_page]
            parallel = inputs[gallery_parallel]
            gallery_size = shared.options.ui.gallery_size

            return {gallery: {"value": project_obj.list_all_frame_paths(parallel)[(page - 1) * gallery_size:page * gallery_size]}}

        def render_video(inputs: CallbackInputs, is_final: bool) -> Iterator[CallbackOutputs]:
            yield {
                render_draft: {"interactive": False},
                render_final: {"interactive": False},
            }

            shared.video_renderer = inputs[video_renderer]

            video_path = inputs[stored_project].data.render_video(shared.video_renderer, is_final, inputs[video_parallel_index])
            wait_until(lambda: not video_render_queue.busy)

            yield {
                render_draft: {"interactive": True},
                render_final: {"interactive": True},
                video_preview: {"value": video_path.as_posix()},
            }

        @render_draft.callback("click", [stored_project, video_renderer, video_parallel_index], [render_draft, render_final, video_preview])
        def _(inputs: CallbackInputs) -> Iterator[CallbackOutputs]:
            yield from render_video(inputs, False)

        @render_final.callback("click", [stored_project, video_renderer, video_parallel_index], [render_draft, render_final, video_preview])
        def _(inputs: CallbackInputs) -> Iterator[CallbackOutputs]:
            yield from render_video(inputs, True)

        @render_graphs.callback("click", [stored_project, measuring_parallel_index], [graph_gallery])
        def _(inputs: CallbackInputs) -> CallbackOutputs:
            return {graph_gallery: {"value": [
                x.plot(inputs[measuring_parallel_index] - 1)
                for x in inputs[stored_project].data.pipeline.modules.values()
                if isinstance(x, MeasuringModule) and x.enabled
            ]}}

        @delete_intermediate_frames.callback("click", [stored_project], [description, gallery])
        def _(inputs: CallbackInputs) -> CallbackOutputs:
            project_obj = inputs[stored_project].data
            project_obj.delete_intermediate_frames()

            return {
                description: {"value": project_obj.get_description()},
                gallery: {"value": project_obj.list_all_frame_paths()[:shared.options.ui.gallery_size]},
            }

        @delete_session_data.callback("click", [stored_project], [])
        def _(inputs: CallbackInputs) -> CallbackOutputs:
            project_obj = inputs[stored_project].data
            project_obj.delete_session_data()
            project_obj.save(project_obj.path)

            return {}

        @apply_settings.callback("click", [options], [])
        def _(inputs: CallbackInputs) -> CallbackOutputs:
            shared.options = inputs[options]
            shared.options.save(EXTENSION_DIR / "settings")

            return {}

        return self._ui.finalize(stored_project, load_parameters, continue_from_last_frame, iter_count, project)

    def run(self, p: StableDiffusionProcessingImg2Img, *args: Any) -> Any:
        stored_project: FSStoreListEntry[Project]
        load_parameters: bool
        continue_from_last_frame: bool
        iter_count: int
        project: Project

        stored_project, load_parameters, continue_from_last_frame, iter_count, project = self._ui.recombine(*args)

        opts_backup = opts.data.copy()

        opts.save_to_dirs = False

        if shared.options.live_preview.show_only_finished_images:
            opts.show_progress_every_n_steps = -1

        p.prompt = prompt_styles.apply_styles_to_prompt(p.prompt, p.styles)
        p.negative_prompt = prompt_styles.apply_negative_styles_to_prompt(p.negative_prompt, p.styles)
        p.styles.clear()

        fix_seed(p)

        project.path = stored_project.data.path
        project.options = opts
        project.processing = p
        project.controlnet_units = get_cn_units(p)

        if load_parameters:
            project.load(stored_project.data.path)

        if not continue_from_last_frame:
            project.delete_all_frames()
            project.delete_session_data()

        if not p.init_images or not isinstance(p.init_images[0], PILImage):
            noises = [
                project.initial_noise.noise.generate((p.height, p.width, 3), p.seed, i)
                for i in range(project.pipeline.parallel)
            ]

            if project.initial_noise.factor < 1.0:
                if not (processed_images := process_images(
                    copy_with_overrides(p,
                        denoising_strength = 1.0 - project.initial_noise.factor,
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

        elif len(p.init_images) != project.pipeline.parallel:
            p.init_images = [p.init_images[0]] * project.pipeline.parallel

        if not project.iteration.images:
            project.iteration.images[:] = [pil_to_np(ensure_image_dims(x, "RGB", (p.width, p.height))) for x in p.init_images]

        last_images = project.iteration.images.copy()

        state.job_count = iter_count

        for i in range(iter_count):
            logging.info(f"Iteration {i + 1} / {iter_count}")

            start_time = perf_counter()

            state.job = "Temporal main loop"
            state.job_no = i

            if not project.pipeline.run(project):
                break

            last_images = project.iteration.images.copy()

            if i % shared.options.output.autosave_every_n_iterations == 0:
                project.save(project.path)

            end_time = perf_counter()

            logging.info(f"Iteration took {end_time - start_time:.6f} second(s)")

        project.pipeline.finalize(project)
        project.save(project.path)

        state.end()

        opts.data.update(opts_backup)

        return Processed(p, [np_to_pil(x) for x in last_images])
