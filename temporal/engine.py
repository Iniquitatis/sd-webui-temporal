import dataclasses
from time import perf_counter

import numpy as np

from temporal.project import Project
from temporal.shared import shared
from temporal.utils import logging
from temporal.utils.image import ensure_image_dims


class Engine:
    def start(self, project: Project, iter_count: int) -> None:
        if project.general.initial_image is None:
            project.general.initial_image = np.full((512, 512, 3), 0.5)

        if project.iteration.image is None:
            project.iteration.image = project.general.initial_image.copy()

        project.iteration.image = ensure_image_dims(project.iteration.image, (project.general.image_size.x, project.general.image_size.y), 3)

        with shared.state_lock:
            shared.state = dataclasses.replace(
                shared.state,
                # FIXME: Shouldn't be controlled by this class
                active_project = project,
                running = True,
                state = "active",
                current_iteration = 0,
                total_iterations = iter_count,
                preview = project.iteration.image,
            )

        for i in range(iter_count):
            with shared.state_lock:
                if not shared.state.running:
                    shared.state.state = "stopping"
                    break

            logging.info(f"Iteration {i + 1} / {iter_count}")

            with shared.state_lock:
                shared.state.current_iteration = i

            start_time = perf_counter()

            if not project.pipeline.run(project.general, project.iteration):
                break

            if i % shared.settings.execution.autosave_every_nth_iteration == 0:
                project.save(project.general.path)

            end_time = perf_counter()

            logging.info(f"Iteration took {end_time - start_time:.6f} second(s)")

        project.pipeline.finalize(project.general, project.iteration)

        project.save(project.general.path)

        with shared.state_lock:
            shared.state = dataclasses.replace(
                shared.state,
                # FIXME: Shouldn't be controlled by this class
                active_project = Project(),
                running = False,
                state = "stopped",
                current_iteration = 0,
                total_iterations = 0,
                preview = None,
            )

    def stop(self) -> None:
        shared.backend.interrupt()

        with shared.state_lock:
            shared.state.running = False
