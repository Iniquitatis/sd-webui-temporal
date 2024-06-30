from pathlib import Path
from typing import Any, Iterator

from temporal.compat import VERSION, upgrade_project
from temporal.meta.serializable import Serializable, SerializableField as Field
from temporal.session import IterationData, Session
from temporal.utils import logging
from temporal.utils.fs import clear_directory, ensure_directory_exists, remove_entry
from temporal.video_renderer import VideoRenderer


FRAME_NAME_FORMAT = "{index:05d}"
FRAME_EXTENSION = "png"
VIDEO_NAME_FORMAT = "{name}-{suffix}"
VIDEO_EXTENSION = "mp4"
VIDEO_DRAFT_SUFFIX = "draft"
VIDEO_FINAL_SUFFIX = "final"


def make_frame_name(index: int) -> str:
    return FRAME_NAME_FORMAT.format(index = index)


def make_frame_file_name(index: int) -> str:
    return f"{make_frame_name(index)}.{FRAME_EXTENSION}"


def make_video_name(name: str, is_final: bool) -> str:
    return VIDEO_NAME_FORMAT.format(
        name = name,
        suffix = VIDEO_FINAL_SUFFIX if is_final else VIDEO_DRAFT_SUFFIX,
    )


def make_video_file_name(name: str, is_final: bool) -> str:
    return f"{make_video_name(name, is_final)}.{VIDEO_EXTENSION}"


def render_project_video(project_dir: Path, renderer: VideoRenderer, is_final: bool, parallel_index: int = 1) -> Path:
    project = Project(project_dir)
    video_path = ensure_directory_exists(project_dir / "videos") / make_video_file_name(f"{parallel_index:02d}", is_final)
    renderer.enqueue_video_render(video_path, project.list_all_frame_paths(parallel_index), is_final)
    return video_path


class Project(Serializable):
    path: Path = Field(Path("outputs/temporal/untitled"), saved = False)
    name: str = Field("untitled", saved = False)
    version: int = Field(VERSION)
    session: Session = Field(factory = Session)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.session.project = self

    def load(self, dir: Path) -> None:
        upgrade_project(dir)
        super().load(dir / "project")

    def save(self, dir: Path) -> None:
        super().save(dir / "project")

    def get_description(self) -> str:
        return "\n\n".join(f"{k}: {v}" for k, v in {
            "Name": self.name,
            "Prompt": self.session.processing.prompt,
            "Negative prompt": self.session.processing.negative_prompt,
            "Checkpoint": self.session.options.sd_model_checkpoint,
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
        clear_directory(self.path, f"*.{FRAME_EXTENSION}")

    def delete_intermediate_frames(self) -> None:
        kept_indices = self.get_first_frame_index(), self.get_last_frame_index()

        for image_path in self._iterate_frame_paths():
            frame_index, _ = _parse_frame_index(image_path)

            if frame_index not in kept_indices:
                remove_entry(image_path)

    def delete_session_data(self) -> None:
        for module in self.session.pipeline.modules.values():
            module.reset()

        self.session.iteration = IterationData()

    def _iterate_frame_paths(self) -> Iterator[Path]:
        return self.path.glob(f"*.{FRAME_EXTENSION}")


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
