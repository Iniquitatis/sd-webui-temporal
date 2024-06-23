from io import BytesIO
from json import loads
from pathlib import Path
from subprocess import run
from typing import Literal, Optional

from PIL import Image

from temporal.meta.serializable import Archive, Serializable, SerializableField as Field
from temporal.utils.fs import ensure_directory_exists
from temporal.utils.image import NumpyImage, pil_to_np


class Video(Serializable):
    path: Optional[Path] = Field(None, saved = False)
    format: Literal["unknown", "avi", "mkv", "mp4"] = Field("unknown")
    data: Optional[bytes] = Field(None)

    @classmethod
    def load_from_path(cls, path: Path) -> "Video":
        return Video(
            format = path.suffix[1:],
            data = path.read_bytes(),
        )

    def write(self, ar: Archive) -> None:
        self.store_in_memory()
        super().write(ar)

    def store_in_memory(self) -> Optional[bytes]:
        if self.path is None:
            return self.data

        self.data = self.path.read_bytes()
        self.path.unlink()
        self.path = None

        return self.data

    def store_on_disk(self) -> Optional[Path]:
        if self.data is None:
            return self.path

        self.path = ensure_directory_exists(Path("tmp")) / f"{id(self)}.{self.format}"
        self.path.write_bytes(self.data)
        self.data = None

        return self.path

    def get_frame_count(self) -> int:
        if (path := self.store_on_disk()) is None:
            return 0

        output = run([
            "ffprobe",
            "-loglevel", "fatal",
            "-print_format", "json",
            "-show_streams",
            "-select_streams", "v",
            "-count_frames",
            path,
        ], capture_output = True)

        try:
            return int(loads(output.stdout.decode())["streams"][0]["nb_frames"])
        except:
            return 0

    def get_frame(self, index: int) -> Optional[NumpyImage]:
        if (path := self.store_on_disk()) is None:
            return None

        output = run([
            "ffmpeg",
            "-i", path,
            "-vf", f"select=eq(n\\,{index})",
            "-vframes", "1",
            "-c:v", "png",
            "-f", "image2pipe",
            "-",
        ], capture_output = True)

        try:
            return pil_to_np(Image.open(BytesIO(output.stdout)))
        except:
            return None
