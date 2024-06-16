import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterator, Optional

from temporal.compat import VERSION, upgrade_project
from temporal.utils import logging
from temporal.utils.fs import clear_directory, ensure_directory_exists, is_directory_empty, remove_directory, remove_entry, save_text
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
        def remove_file(elem: Optional[ET.Element]) -> None:
            if elem is None:
                return

            if elem.text is not None:
                remove_entry(self.session_path / elem.text)

            elem.set("type", "NoneType")
            elem.text = ""

        if not (session_data_path := (self.session_path / "data.xml")).exists():
            return

        tree = ET.ElementTree(file = session_data_path)
        root = tree.getroot()

        if (measurements := root.find("*[@key='pipeline']/*[@key='modules']/*[@key='measuring']/*[@key='metrics']/*[@key='measurements']")) is not None:
            for measurement in reversed(measurements):
                measurements.remove(measurement)

        remove_file(root.find("*[@key='pipeline']/*[@key='modules']/*[@key='averaging']/*[@key='buffer']"))
        remove_file(root.find("*[@key='pipeline']/*[@key='modules']/*[@key='interpolation']/*[@key='buffer']"))
        remove_file(root.find("*[@key='pipeline']/*[@key='modules']/*[@key='limiting']/*[@key='buffer']"))

        if (last_index := root.find("*[@key='pipeline']/*[@key='modules']/*[@key='averaging']/*[@key='last_index']")) is not None:
            last_index.text = "0"

        if (iteration := root.find("*[@key='iteration']")) is not None:
            if (images := iteration.find("*[@key='images']")):
                for image in reversed(images):
                    remove_file(image)
                    images.remove(image)

            if (index := iteration.find("*[@key='index']")) is not None:
                index.text = "1"

            if (module_id := iteration.find("*[@key='module_id']")) is not None:
                module_id.set("type", "NoneType")
                module_id.text = ""

        ET.indent(tree)
        tree.write(session_data_path)

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
