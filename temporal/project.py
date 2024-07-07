from pathlib import Path
from typing import TYPE_CHECKING, Iterator, Optional

from temporal.backend import BackendData
from temporal.compat import get_latest_version, upgrade_project
from temporal.meta.serializable import Serializable, SerializableField as Field
from temporal.noise import Noise
if TYPE_CHECKING:
    from temporal.pipeline import Pipeline
from temporal.utils import logging
from temporal.utils.fs import clear_directory, ensure_directory_exists, remove_entry
from temporal.utils.image import NumpyImage
from temporal.video_renderer import VideoRenderer


class InitialNoiseParams(Serializable):
    factor: float = Field(0.0)
    noise: Noise = Field(factory = Noise)


class IterationData(Serializable):
    images: list[NumpyImage] = Field(factory = list)
    index: int = Field(1)
    module_id: Optional[str] = Field(None)


class Project(Serializable):
    path: Path = Field(Path("outputs/temporal/untitled"), saved = False)
    version: int = Field(get_latest_version())
    backend_data: BackendData = Field(factory = BackendData)
    initial_noise: InitialNoiseParams = Field(factory = InitialNoiseParams)
    pipeline: "Pipeline" = Field(factory = lambda: _make_pipeline())
    iteration: IterationData = Field(factory = IterationData)

    def load(self, dir: Path) -> None:
        upgrade_project(dir)
        super().load(dir / "project")

    def save(self, dir: Path) -> None:
        super().save(dir / "project")

    def get_description(self) -> str:
        return "\n\n".join(f"{k}: {v}" for k, v in {
            "Name": self.path.name,
            "Positive prompt": self.backend_data.positive_prompt,
            "Negative prompt": self.backend_data.negative_prompt,
            "Model": self.backend_data.model,
            "Last frame": self.get_last_frame_index(),
            "Saved frames": self.get_actual_frame_count(),
        }.items())

    def get_first_frame_index(self) -> int:
        return min((_parse_frame_index(x)[0] for x in self._iterate_frame_paths()), default = 0)

    def get_last_frame_index(self) -> int:
        return max((_parse_frame_index(x)[0] for x in self._iterate_frame_paths()), default = 0)

    def get_actual_frame_count(self, parallel_index: int = 1) -> int:
        return sum(_parse_frame_index(x)[1] == parallel_index for x in self._iterate_frame_paths())

    def list_all_frame_paths(self, parallel_index: int = 1) -> list[Path]:
        return sorted((x for x in self._iterate_frame_paths() if _parse_frame_index(x)[1] == parallel_index), key = lambda x: x.name)

    def delete_all_frames(self) -> None:
        clear_directory(self.path, "*.png")

    def delete_intermediate_frames(self) -> None:
        kept_indices = self.get_first_frame_index(), self.get_last_frame_index()

        for image_path in self._iterate_frame_paths():
            frame_index, _ = _parse_frame_index(image_path)

            if frame_index not in kept_indices:
                remove_entry(image_path)

    def delete_session_data(self) -> None:
        for module in self.pipeline.modules:
            module.reset()

        self.iteration = IterationData()

    def render_video(self, renderer: VideoRenderer, is_final: bool, parallel_index: int = 1) -> Path:
        video_path = ensure_directory_exists(self.path / "videos") / f"{parallel_index:02d}-{'final' if is_final else 'draft'}.mp4"
        renderer.enqueue_video_render(video_path, self.list_all_frame_paths(parallel_index), is_final)
        return video_path

    def _iterate_frame_paths(self) -> Iterator[Path]:
        return self.path.glob("*.png")


def _make_pipeline() -> "Pipeline":
    from temporal.pipeline import Pipeline
    return Pipeline()


def _parse_frame_index(image_path: Path) -> tuple[int, int]:
    if image_path.is_file():
        try:
            return int(image_path.stem), 1
        except:
            pass

        try:
            frame_index, parallel_index = image_path.stem.split("-")
            return int(frame_index), int(parallel_index)
        except:
            pass

        logging.warning(f"{image_path.stem} doesn't match the frame name format")

    return 0, 0
