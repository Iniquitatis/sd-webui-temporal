from dataclasses import dataclass, replace
from threading import Lock
from time import perf_counter
from typing import Literal, Optional

import numpy as np

from temporal.project import Project
from temporal.shared import shared
from temporal.utils.image import NumpyImage, ensure_image_dims
from temporal.utils.logging import log


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

    def start(self, project: Project, iter_count: int) -> None:
        if project.general.initial_image is None:
            project.general.initial_image = np.full((512, 512, 3), 0.5)

        if project.iteration.image is None:
            project.iteration.image = project.general.initial_image.copy()

        project.iteration.image = ensure_image_dims(project.iteration.image, (project.general.image_size.x, project.general.image_size.y), 3)

        with self.state_lock:
            self.state = replace(
                self.state,
                running = True,
                state = "active",
                current_iteration = 0,
                total_iterations = iter_count,
                preview = project.iteration.image,
            )

        for i in range(iter_count):
            with self.state_lock:
                if not self.state.running:
                    self.state.state = "stopping"
                    break

            log.info(f"Iteration {i + 1} / {iter_count}")

            with self.state_lock:
                self.state.current_iteration = i

            start_time = perf_counter()

            if project.general.mode == "loop":
                unprocessed_image = project.general.initial_image
            elif project.general.mode == "recursion":
                unprocessed_image = project.iteration.image
            else:
                raise ValueError

            success = True

            for j, image, preview in project.pipeline.run(unprocessed_image, project.general, project.iteration.index):
                if j < project.iteration.step:
                    continue

                if not self.state.running or image is None:
                    success = False
                    break

                project.iteration.image = image
                project.iteration.step += 1

                if preview:
                    self.state.preview = project.iteration.image

            if not success:
                break

            project.iteration.index += 1
            project.iteration.step = 0

            if i % shared.settings.execution.autosave_every_nth_iteration == 0:
                project.save(project.general.path)

            end_time = perf_counter()

            log.info(f"Iteration took {end_time - start_time:.6f} second(s)")

        project.pipeline.finalize(project.iteration.image, project.general)

        project.save(project.general.path)

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
        shared.backend.interrupt()

        with self.state_lock:
            self.state.running = False
