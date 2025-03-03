from pathlib import Path
from random import randint
from typing import Any, Iterator, Optional

from temporal.meta.serializable import Serializable, SerializableField as Field
from temporal.noise import Noise
from temporal.utils import logging
from temporal.utils.fs import clear_directory, ensure_directory_exists, remove_entry
from temporal.utils.image import NumpyImage, load_image, pil_to_np
from temporal.vector import IntVector
from temporal.video_renderer import VideoRenderer


class GeneralData(Serializable):
    # FIXME: Path should be constructed dynamically by getting the global
    # project directory and the name (sanitized, of course)
    path: Path = Field(Path("outputs/temporal/untitled"), flags = {"runtime"})
    name: str = Field("untitled")
    description: str = Field("")
    initial_image: Optional[NumpyImage] = Field(None, variant = "image")
    initial_noise: Noise = Field(factory = Noise)
    image_size: IntVector = Field(factory = lambda: IntVector(512, 512))
    seed: int = Field(-1)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        if self.seed == -1:
            self.seed = randint(0, 0x7fffffff)

    def get_first_frame_index(self) -> int:
        return min((_parse_frame_index(x) for x in self._iterate_frame_paths()), default = 0)

    def get_last_frame_index(self) -> int:
        return max((_parse_frame_index(x) for x in self._iterate_frame_paths()), default = 0)

    def get_actual_frame_count(self) -> int:
        return sum(1 for _ in self._iterate_frame_paths())

    def list_all_frame_paths(self) -> list[Path]:
        return sorted((x for x in self._iterate_frame_paths()), key = lambda x: x.name)

    def get_last_frame(self) -> Optional[NumpyImage]:
        if index := self.get_last_frame_index():
            return pil_to_np(load_image(self.path / f"{index:05d}.png"))

    def delete_all_frames(self) -> None:
        clear_directory(self.path, "*.png")

    def delete_intermediate_frames(self) -> None:
        kept_indices = self.get_first_frame_index(), self.get_last_frame_index()

        for image_path in self._iterate_frame_paths():
            if _parse_frame_index(image_path) not in kept_indices:
                remove_entry(image_path)

    def render_video(self, renderer: VideoRenderer, is_final: bool, enqueue: bool = True) -> Path:
        # FIXME
        video_path = ensure_directory_exists(self.path / "videos") / f"{'final' if is_final else 'draft'}.mp4"

        if enqueue:
            method = renderer.enqueue_video_render
        else:
            method = renderer._render_video

        method(video_path, self.list_all_frame_paths(), is_final)

        return video_path

    def _iterate_frame_paths(self) -> Iterator[Path]:
        return self.path.glob("*.png")


def _parse_frame_index(image_path: Path) -> int:
    if image_path.is_file():
        try:
            return int(image_path.stem)
        except:
            logging.warning(f"{image_path.stem} doesn't match the frame name format")

    return 0
