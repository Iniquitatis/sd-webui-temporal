from typing import Iterator

import gradio as gr

from temporal.project import Project
from temporal.project_store import ProjectStore
from temporal.shared import shared
from temporal.ui import Callback, CallbackInputs, CallbackOutputs, ReadData, UIThing, UpdateData, UpdateRequest, Widget
from temporal.ui.fs_store_list import FSStoreList
from temporal.ui.gradio_widget import GradioWidget
from temporal.ui.initial_noise_editor import InitialNoiseEditor
from temporal.ui.paginator import Paginator
from temporal.ui.pipeline_editor import PipelineEditor


class ProjectEditor(Widget):
    def __init__(
        self,
        store: ProjectStore,
        value: Project = Project(),
    ) -> None:
        super().__init__()

        self._project = FSStoreList(label = "Project", store = store, features = ["load", "rename", "delete"], value = value)

        with GradioWidget(gr.Tab, label = "Information"):
            self._description = GradioWidget(gr.Textbox, label = "Description", lines = 5, max_lines = 5, interactive = False)
            self._gallery = GradioWidget(gr.Gallery, label = "Gallery", columns = 4, object_fit = "contain", preview = True)
            self._gallery_page = Paginator(label = "Page", minimum = 1, value = 1)
            self._gallery_parallel = Paginator(label = "Parallel", minimum = 1, value = 1)

        with GradioWidget(gr.Tab, label = "Tools"):
            self._delete_intermediate_frames = GradioWidget(gr.Button, value = "Delete intermediate frames")
            self._delete_session_data = GradioWidget(gr.Button, value = "Delete session data")

        with GradioWidget(gr.Tab, label = "Pipeline"):
            self._initial_noise = InitialNoiseEditor(value.session.initial_noise)
            self._pipeline = PipelineEditor(value.session.pipeline)

        @self._project.callback("change", [self._project], [self._description, self._gallery, self._gallery_page, self._gallery_parallel, self._initial_noise, self._pipeline])
        def _(inputs: CallbackInputs) -> CallbackOutputs:
            project = inputs[self._project]

            return {
                self._description: {"value": project.get_description()},
                self._gallery: {"value": project.list_all_frame_paths()[:shared.options.ui.gallery_size]},
                self._gallery_page: {"value": 1},
                self._gallery_parallel: {"value": 1},
                self._initial_noise: {"value": project.session.initial_noise},
                self._pipeline: {"value": project.session.pipeline},
            }

        @self._project.callback("load", [self._project], [self._initial_noise, self._pipeline])
        def _(inputs: CallbackInputs) -> CallbackOutputs:
            project = inputs[self._project]

            return {
                self._initial_noise: {"value": project.session.initial_noise},
                self._pipeline: {"value": project.session.pipeline},
            }

        def update_gallery(inputs: CallbackInputs) -> CallbackOutputs:
            project = inputs[self._project]
            page = inputs[self._gallery_page]
            parallel = inputs[self._gallery_parallel]
            gallery_size = shared.options.ui.gallery_size

            return {self._gallery: {"value": project.list_all_frame_paths(parallel)[(page - 1) * gallery_size:page * gallery_size]}}

        @self._gallery_page.callback("change", [self._project, self._gallery_page, self._gallery_parallel], [self._gallery])
        def _(inputs: CallbackInputs) -> CallbackOutputs:
            return update_gallery(inputs)

        @self._gallery_parallel.callback("change", [self._project, self._gallery_page, self._gallery_parallel], [self._gallery])
        def _(inputs: CallbackInputs) -> CallbackOutputs:
            return update_gallery(inputs)

        @self._delete_intermediate_frames.callback("click", [self._project], [self._description, self._gallery])
        def _(inputs: CallbackInputs) -> CallbackOutputs:
            project = inputs[self._project]
            project.delete_intermediate_frames()

            return {
                self._description: {"value": project.get_description()},
                self._gallery: {"value": project.list_all_frame_paths()[:shared.options.ui.gallery_size]},
            }

        @self._delete_session_data.callback("click", [self._project], [])
        def _(inputs: CallbackInputs) -> CallbackOutputs:
            project = inputs[self._project]
            project.delete_session_data()
            project.save(project.path)

            return {}

    @property
    def dependencies(self) -> Iterator[UIThing]:
        yield self._project
        yield self._initial_noise
        yield self._pipeline

    def read(self, data: ReadData) -> Project:
        project = data[self._project]
        project.session.initial_noise = data[self._initial_noise]
        project.session.pipeline = data[self._pipeline]
        return project

    def update(self, data: UpdateData) -> UpdateRequest:
        result: UpdateRequest = {}

        if isinstance(value := data.get("value", None), Project):
            result[self._project] = {"name": value.name, "value": value}
            result[self._initial_noise] = {"value": value.session.initial_noise}
            result[self._pipeline] = {"value": value.session.pipeline}

        return result

    def setup_callback(self, callback: Callback) -> None:
        self._project.setup_callback(callback)
