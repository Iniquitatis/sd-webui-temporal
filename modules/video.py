from io import BytesIO
from json import loads
from pathlib import Path
from subprocess import run
from typing import Literal, Optional
from typing_extensions import Self

from PIL import Image

from modules.object import Field, Object
from modules.serialization import JSONValue, SerializationParams
from modules.utils.base64 import decode_with_mime_type, encode_with_mime_type
from modules.utils.fs import ensure_directory_exists
from modules.utils.image import NumpyImage, pil_to_np


class Video(Object):
    path: Optional[Path] = Field(None, flags = {"runtime"})
    format: Literal["unknown", "avi", "mkv", "mp4"] = Field("unknown")
    data: Optional[bytes] = Field(None)

    @classmethod
    def load_from_path(cls, path: Path) -> Self:
        return cls(
            format = path.suffix[1:],
            data = path.read_bytes(),
        )

    @classmethod
    def from_json(cls, data: JSONValue, params: SerializationParams = SerializationParams()) -> Self:
        if params.data_dir is not None:
            return super().from_json(data, params)
        elif isinstance(data, str):
            type, subtype, video_data = decode_with_mime_type(data)

            if type != "video":
                raise ValueError

            return cls(
                format = subtype,
                data = video_data,
            )
        else:
            raise ValueError

    def to_json(self, params: SerializationParams = SerializationParams()) -> JSONValue:
        self.store_in_memory()

        if params.data_dir is not None:
            return super().to_json(params)
        elif self.data is not None:
            return encode_with_mime_type("video", self.format, self.data)
        else:
            return None

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
