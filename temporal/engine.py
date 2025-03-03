from dataclasses import dataclass
from threading import Lock
from time import perf_counter
from typing import Literal, Optional

from temporal.project import Project
from temporal.shared import shared
from temporal.utils import logging
from temporal.utils.image import ensure_image_dims


@dataclass
class ExecutionState:
    active_project: Optional[Project] = None
    running: bool = False
    state: Literal["active", "stopped"] = "stopped"
    current_iteration: int = 0
    total_iterations: int = 0


class Engine:
    def __init__(self) -> None:
        self.state = ExecutionState()
        self._state_lock = Lock()

    def start(self, project: Project, iter_count: int) -> None:
        with self._state_lock:
            self.state = ExecutionState(project, True, "active", 0, iter_count)

        if project.general.initial_image is None:
            project.general.initial_image = project.general.initial_noise.generate((project.general.image_size.y, project.general.image_size.x, 3), project.general.seed)

        if project.iteration.image is None:
            project.iteration.image = project.general.initial_image.copy()

        project.iteration.image = ensure_image_dims(project.iteration.image, (project.general.image_size.x, project.general.image_size.y), 3)

        for i in range(iter_count):
            with self._state_lock:
                if not self.state.running:
                    break

            logging.info(f"Iteration {i + 1} / {iter_count}")

            with self._state_lock:
                self.state.current_iteration = i

            start_time = perf_counter()

            if not project.pipeline.run(project.general, project.iteration):
                break

            if i % shared.options.output.autosave_every_n_iterations == 0:
                project.save(project.general.path)

            end_time = perf_counter()

            logging.info(f"Iteration took {end_time - start_time:.6f} second(s)")

        project.pipeline.finalize(project.general, project.iteration)

        project.save(project.general.path)

        with self._state_lock:
            self.state = ExecutionState()

    def stop(self) -> None:
        shared.backend.interrupt()

        with self._state_lock:
            self.state.running = False
