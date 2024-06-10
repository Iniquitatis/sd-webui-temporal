import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterator, Optional

from temporal.compat import VERSION, upgrade_project
from temporal.utils import logging
from temporal.utils.fs import clear_directory, is_directory_empty, remove_directory, remove_entry, save_text
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


def render_project_video(output_dir: Path, project_subdir: str, renderer: VideoRenderer, is_final: bool) -> None:
    project = Project(output_dir / project_subdir)
    renderer.enqueue_video_render(output_dir / make_video_file_name(project.name, is_final), project.list_all_frame_paths(), is_final)


class Project:
    def __init__(self, path: Path) -> None:
        self.path = path

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def data_path(self) -> Path:
        return self.path / "project"

    @property
    def session_path(self) -> Path:
        return self.data_path / "session"

    def load(self) -> None:
        if is_directory_empty(self.path):
            return

        upgrade_project(self.path)

    def save(self) -> None:
        save_text(self.data_path / "version.txt", str(VERSION))

    def get_description(self) -> Optional[str]:
        if not (path := self.session_path / "data.xml").is_file():
            return None

        tree = ET.ElementTree(file = path)
        values = {
            "Name": self.name,
            "Prompt": tree.findtext("*[@key='processing']/*[@key='prompt']"),
            "Negative prompt": tree.findtext("*[@key='processing']/*[@key='negative_prompt']"),
            "Checkpoint": tree.findtext("*[@key='options']/*[@key='sd_model_checkpoint']"),
            "Last frame": self.get_last_frame_index(),
            "Saved frames": self.get_actual_frame_count(),
        }
        return "\n\n".join(f"{k}: {v}" for k, v in values.items() if v is not None)

    def get_first_frame_index(self) -> int:
        return min((_parse_frame_index(x) for x in self._iterate_frame_paths()), default = 0)

    def get_last_frame_index(self) -> int:
        return max((_parse_frame_index(x) for x in self._iterate_frame_paths()), default = 0)

    def get_actual_frame_count(self) -> int:
        return sum(1 for _ in self._iterate_frame_paths())

    def list_all_frame_paths(self) -> list[Path]:
        return sorted(self._iterate_frame_paths(), key = lambda x: x.name)

    def delete_all_frames(self) -> None:
        clear_directory(self.path, f"*.{FRAME_EXTENSION}")

    def delete_intermediate_frames(self) -> None:
        kept_paths = set()
        min_index = int(1e9)
        max_index = 0

        for image_path in self._iterate_frame_paths():
            if frame_index := _parse_frame_index(image_path):
                min_index = min(min_index, frame_index)
                max_index = max(max_index, frame_index)
            else:
                kept_paths.add(image_path)

        kept_paths.add(self.path / make_frame_file_name(min_index))
        kept_paths.add(self.path / make_frame_file_name(max_index))

        for image_path in self._iterate_frame_paths():
            if image_path not in kept_paths:
                remove_entry(image_path)

    def delete_session_data(self) -> None:
        if not (session_data_path := (self.session_path / "data.xml")).exists():
            return

        tree = ET.ElementTree(file = session_data_path)
        root = tree.getroot()

        if (measurements_elem := root.find("*[@key='modules']/*[@key='measuring']/*[@key='metrics']/*[@key='measurements']")) is not None:
            for child in list(measurements_elem):
                measurements_elem.remove(child)

        if (buffer_elem := root.find("*[@key='modules']/*[@key='averaging']/*[@key='buffer']")) is not None:
            if (array_path := buffer_elem.findtext("*[@key='array']")) is not None:
                remove_entry(Path(array_path))

            root.remove(buffer_elem)

        tree.write(session_data_path)

        remove_directory(self.session_path / "metrics")

    def _iterate_frame_paths(self) -> Iterator[Path]:
        return self.path.glob(f"*.{FRAME_EXTENSION}")


def _parse_frame_index(image_path: Path) -> int:
    if image_path.is_file():
        try:
            return int(image_path.stem)
        except:
            logging.warning(f"{image_path.stem} doesn't match the frame name format")

    return 0