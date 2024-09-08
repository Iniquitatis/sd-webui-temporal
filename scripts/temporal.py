from pathlib import Path
from typing import Any, Iterator

import gradio as gr

from modules import scripts, shared as webui_shared
from modules.options import Options
from modules.processing import Processed, StableDiffusionProcessingImg2Img, fix_seed
from modules.shared_state import State
from modules.styles import StyleDatabase

from temporal.backends.webui import WebUIBackend, WebUIImageToImageParams
from temporal.backends.webui.controlnet import get_controlnet_units
from temporal.engine import Engine
from temporal.pipeline_modules.measuring import MeasuringModule
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
from temporal.utils.fs import load_text
from temporal.utils.image import PILImage, np_to_pil
from temporal.utils.time import wait_until
from temporal.video_renderer import video_render_queue


# FIXME: To shut up the type checker
opts: Options = getattr(webui_shared, "opts")
prompt_styles: StyleDatabase = getattr(webui_shared, "prompt_styles")
state: State = getattr(webui_shared, "state")


EXTENSION_DIR = Path(scripts.basedir())


class WebUIEngine(Engine):
    def on_iteration(self, iteration: int) -> None:
        state.job = "Temporal main loop"
        state.job_no = iteration


class TemporalScript(scripts.Script):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.engine = WebUIEngine(WebUIBackend(), EXTENSION_DIR / "settings", EXTENSION_DIR / "presets")

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
                for x in inputs[stored_project].data.pipeline.modules
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

        fix_seed(p)

        project.path = stored_project.data.path
        project.parameters = WebUIImageToImageParams(
            model = opts.sd_model_checkpoint,
            vae = opts.sd_vae,
            clip_skip = opts.CLIP_stop_at_last_layers,
            images = [x for x in p.init_images if isinstance(x, PILImage)],
            positive_prompts = [p.prompt],
            negative_prompts = [p.negative_prompt],
            width = p.width,
            height = p.height,
            sampler = p.sampler_name,
            scheduler = p.scheduler,
            steps = p.steps,
            cfg = p.cfg_scale,
            strength = p.denoising_strength,
            seeds = [p.seed],
            options = opts,
            processing = p,
            controlnet_units = get_controlnet_units(p),
        )

        if load_parameters:
            project.load(stored_project.data.path)

        if not continue_from_last_frame:
            project.delete_all_frames()
            project.delete_session_data()

        state.job_count = iter_count

        images = self.engine.start(project, iter_count)

        state.end()

        return Processed(p, [np_to_pil(x) for x in images])
