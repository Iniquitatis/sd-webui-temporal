from importlib import import_module
from pathlib import Path
from threading import Lock
from time import perf_counter

from temporal.backend import Backend
from temporal.project import Project
from temporal.shared import shared
from temporal.utils import logging
from temporal.utils.image import NumpyImage, ensure_image_dims
from temporal.utils.object import set_property_by_path


class Engine:
    def __init__(self, backend: Backend, options_path: Path, presets_path: Path) -> None:
        self.active_project = None
        self.running = False
        self.state = "stopped"
        self.current_iteration = 0
        self.total_iterations = 0
        self._state_lock = Lock()

        shared.init(backend, options_path, presets_path)

        for path in (Path(__file__).parent / "pipeline_modules").rglob("*.py"):
            if path.name != "__init__":
                import_module(f"temporal.pipeline_modules.{path.parent.stem}.{path.stem}")

    def on_start(self) -> None:
        pass

    def on_iteration(self, iteration: int) -> None:
        pass

    def on_end(self) -> None:
        pass

    def start(self, project: Project, iter_count: int) -> list[NumpyImage]:
        with self._state_lock:
            self.active_project = project
            self.running = True
            self.state = "active"
            self.total_iterations = iter_count

        if project.general.image is None:
            noises = [
                project.general.initial_noise.generate((project.general.image_size.y, project.general.image_size.x, 3), project.general.seed, i)
                for i in range(project.general.parallel)
            ]

            project.general.image = noises[0]
            project.iteration.images[:] = noises

        project.general.image = ensure_image_dims(project.general.image, (project.general.image_size.x, project.general.image_size.y), 3)

        if not project.iteration.images:
            project.iteration.images[:] = [project.general.image] * project.general.parallel

        last_images = project.iteration.images.copy()

        self.on_start()

        for i in range(iter_count):
            with self._state_lock:
                if not self.running:
                    break

            logging.info(f"Iteration {i + 1} / {iter_count}")

            with self._state_lock:
                self.current_iteration = i

            start_time = perf_counter()

            self.on_iteration(i)

            for path, value in project.animation.evaluate(project.iteration.index).items():
                set_property_by_path(project, path, value)

            if not project.pipeline.run(project.general, project.iteration):
                break

            last_images = project.iteration.images.copy()

            if i % shared.options.output.autosave_every_n_iterations == 0:
                project.save(project.general.path)

            end_time = perf_counter()

            logging.info(f"Iteration took {end_time - start_time:.6f} second(s)")

        project.pipeline.finalize(project.general, project.iteration)

        project.save(project.general.path)

        self.on_end()

        with self._state_lock:
            self.active_project = None
            self.running = False
            self.state = "stopped"
            self.current_iteration = 0
            self.total_iterations = 0

        return last_images

    def stop(self) -> None:
        shared.backend.interrupt()

        with self._state_lock:
            self.running = False
