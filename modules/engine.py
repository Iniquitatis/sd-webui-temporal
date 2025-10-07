import asyncio
from typing import Optional

import numpy as np

from modules.pipeline_state import PipelineState
from modules.project import Project
from modules.stopwatch import Stopwatch
from modules.utils.image import NumpyImage, ensure_image_dims
from modules.utils.logging import log


class Engine:
    def __init__(self) -> None:
        self.running = False
        self.current_iteration = 0
        self.total_iterations = 0
        self.stopwatch = Stopwatch()
        self.project: Optional[Project] = None
        self.preview: Optional[NumpyImage] = None

    async def start(self, project: Project, iterations: int) -> None:
        self.running = True
        self.current_iteration = 0
        self.total_iterations = iterations
        self.stopwatch.reset()
        self.project = project

        if project.general.initial_image is None:
            project.general.initial_image = np.full((512, 512, 3), 0.5)

        last_image = ensure_image_dims(project.general.initial_image.copy(), channels = 3)

        self.preview = last_image

        for i in range(iterations):
            if not self.running:
                break

            log.info(f"Iteration {i + 1} / {iterations}")

            self.current_iteration = i

            with self.stopwatch:
                if (result := await asyncio.to_thread(self._run_pipeline, last_image, project)) is None:
                    break

                if project.general.mode == "recursion":
                    last_image = result

            log.info(f"Iteration took {self.stopwatch.last:.6f} second(s)")

        self.running = False
        self.current_iteration = 0
        self.total_iterations = 0
        self.stopwatch.reset()
        self.project = None
        self.preview = None

    def stop(self) -> None:
        self.running = False

        if self.project is not None:
            self.project.pipeline.interrupt(self.project.general)

    def _run_pipeline(self, image: NumpyImage, project: Project) -> Optional[NumpyImage]:
        for state in project.pipeline.run(image, project.general):
            if not self.running:
                return

            match state:
                case PipelineState.progress():
                    if state.preview:
                        self.preview = state.image

                case PipelineState.finish():
                    if state.preview:
                        self.preview = state.image

                    return state.image

                case PipelineState.fail():
                    log.warning(state.message)
                    return
