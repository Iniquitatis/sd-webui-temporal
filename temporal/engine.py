from importlib import import_module
from pathlib import Path
from time import perf_counter

from temporal.backend import Backend
from temporal.project import Project
from temporal.shared import shared
from temporal.utils import logging
from temporal.utils.image import NumpyImage, ensure_image_dims
from temporal.utils.object import copy_with_overrides, set_property_by_path
from temporal.utils.prompt import evaluate_prompt


class Engine:
    def __init__(self, backend: Backend, options_path: Path, presets_path: Path) -> None:
        self.state = "stopped"
        self.current_iteration = 0
        self.total_iterations = 0

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
        self.state = "active"
        self.total_iterations = iter_count

        if not project.parameters.images:
            noises = [
                project.initial_noise.noise.generate((project.parameters.height, project.parameters.width, 3), project.parameters.seed, i)
                for i in range(project.pipeline.parallel)
            ]

            if project.initial_noise.factor < 1.0:
                if not (processed_images := shared.backend.images_to_batches(
                    copy_with_overrides(project.parameters,
                        positive_prompts = [
                            evaluate_prompt(x, 0)
                            for x in project.parameters.positive_prompts
                        ],
                        negative_prompts = [
                            evaluate_prompt(x, 0)
                            for x in project.parameters.negative_prompts
                        ],
                        strength = 1.0 - project.initial_noise.factor,
                    ),
                    [(x, project.parameters.seed + i, 1) for i, x in enumerate(noises)],
                    shared.options.processing.pixels_per_batch,
                    True,
                )):
                    return []

                project.parameters.images.clear()
                project.parameters.images[:] = [image_array[0] for image_array in processed_images]

            else:
                project.parameters.images[:] = [x for x in noises]

        elif len(project.parameters.images) != project.pipeline.parallel:
            project.parameters.images[:] = [project.parameters.images[0]] * project.pipeline.parallel

        if not project.iteration.images:
            project.iteration.images[:] = [ensure_image_dims(x, (project.parameters.width, project.parameters.height), 3) for x in project.parameters.images]

        last_images = project.iteration.images.copy()

        self.on_start()

        for i in range(iter_count):
            logging.info(f"Iteration {i + 1} / {iter_count}")

            self.current_iteration = i

            start_time = perf_counter()

            self.on_iteration(i)

            for path, value in project.animation.evaluate(project.iteration.index).items():
                set_property_by_path(project, path, value)

            if not project.pipeline.run(project):
                break

            last_images = project.iteration.images.copy()

            if i % shared.options.output.autosave_every_n_iterations == 0:
                project.save(project.path)

            end_time = perf_counter()

            logging.info(f"Iteration took {end_time - start_time:.6f} second(s)")

        project.pipeline.finalize(project)
        project.save(project.path)

        self.on_end()

        self.state = "stopped"
        self.current_iteration = 0
        self.total_iterations = 0

        return last_images
