from dataclasses import dataclass, replace
from threading import Lock
from time import perf_counter
from typing import Literal, Optional

import numpy as np

from modules.pipeline_state import PipelineState
from modules.project import Project
from modules.utils.image import NumpyImage, ensure_image_dims
from modules.utils.logging import log


@dataclass
class ExecutionState:
    running: bool = False
    state: Literal["active", "stopping", "stopped"] = "stopped"
    current_iteration: int = 0
    total_iterations: int = 0
    preview: Optional[NumpyImage] = None


class Engine:
    def __init__(self) -> None:
        self.state = ExecutionState()
        self.state_lock = Lock()
        self.project: Optional[Project] = None

    def start(self, project: Project, iterations: int) -> None:
        self.project = project

        if project.general.initial_image is None:
            project.general.initial_image = np.full((512, 512, 3), 0.5)

        last_image = ensure_image_dims(project.general.initial_image.copy(), channels = 3)

        with self.state_lock:
            self.state = replace(
                self.state,
                running = True,
                state = "active",
                current_iteration = 0,
                total_iterations = iterations,
                preview = last_image,
            )

        for i in range(iterations):
            with self.state_lock:
                if not self.state.running:
                    self.state.state = "stopping"
                    break

            log.info(f"Iteration {i + 1} / {iterations}")

            with self.state_lock:
                self.state.current_iteration = i

            start_time = perf_counter()

            success = True

            for state in project.pipeline.run(last_image, project.general):
                if not self.state.running:
                    success = False
                    break

                match state:
                    case PipelineState.progress():
                        if state.preview:
                            self.state.preview = state.image

                    case PipelineState.finish():
                        if state.preview:
                            self.state.preview = state.image

                        if project.general.mode == "recursion":
                            last_image = state.image

                    case PipelineState.fail():
                        log.warning(state.message)
                        success = False
                        break

            if not success:
                break

            end_time = perf_counter()

            log.info(f"Iteration took {end_time - start_time:.6f} second(s)")

        project.pipeline.finalize(project.general)

        self.project = None

        with self.state_lock:
            self.state = replace(
                self.state,
                running = False,
                state = "stopped",
                current_iteration = 0,
                total_iterations = 0,
                preview = None,
            )

    def stop(self) -> None:
        if self.project is not None:
            self.project.pipeline.interrupt(self.project.general)

        with self.state_lock:
            self.state.running = False
