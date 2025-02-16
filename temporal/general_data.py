from pathlib import Path
from typing import Annotated, Iterator

import numpy as np

from temporal.meta.serializable import Serializable, SerializableField as Field
from temporal.noise import Noise
from temporal.serialization import Variant
from temporal.utils import logging
from temporal.utils.fs import clear_directory, ensure_directory_exists, remove_entry
from temporal.utils.image import NumpyImage
from temporal.vector import IntVector
from temporal.video_renderer import VideoRenderer


class GeneralData(Serializable):
    path: Path = Field(Path("outputs/temporal/untitled"), saved = False)
    image: Annotated[NumpyImage, Variant("image")] = Field(factory = lambda: np.array([]))
    initial_noise: Noise = Field(factory = Noise)
    image_size: IntVector = Field(factory = lambda: IntVector(512, 512))
    parallel: int = Field(1)
    seed: int = Field(-1)

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

    def render_video(self, renderer: VideoRenderer, is_final: bool, parallel_index: int = 1, enqueue: bool = True) -> Path:
        # FIXME
        video_path = ensure_directory_exists(self.path / "videos") / f"{parallel_index:02d}-{'final' if is_final else 'draft'}.mp4"

        if enqueue:
            method = renderer.enqueue_video_render
        else:
            method = renderer._render_video

        method(video_path, self.list_all_frame_paths(parallel_index), is_final)

        return video_path

    def _iterate_frame_paths(self) -> Iterator[Path]:
        return self.path.glob("*.png")


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
