from itertools import chain
from subprocess import run

from modules.general_data import GeneralData
from modules.object import Field
from modules.pipeline_modules.tool import ToolModule
from modules.utils.fs import ensure_directory_exists, save_text
from modules.utils.image import NumpyImage
from modules.video import Video
from modules.video_filter import VideoFilter


class VideoRenderingModule(ToolModule):
    name = "Video rendering"
    is_visualizable = True
    visualization_type = "video"

    file_name: str = Field("video", name = "File name", display = "box")
    image_name_prefix: str = Field("", name = "Image name prefix", display = "box")
    fps: int = Field(30, name = "FPS", minimum = 1, maximum = 60, step = 1, display = "slider")
    first_frame: int = Field(1, name = "First frame", minimum = 1, step = 1, display = "box")
    last_frame: int = Field(0, name = "Last frame", minimum = 0, step = 1, display = "box")
    frame_stride: int = Field(1, name = "Frame stride", minimum = 1, step = 1, display = "box")
    looping: bool = Field(False, name = "Looping")
    archive_mode: bool = Field(False, name = "Archive mode")
    filters: list[VideoFilter] = Field(list, name = "Filters")
    iteration: int = Field(0, flags = {"runtime"})

    def process(self, image: NumpyImage, general: GeneralData) -> None:
        self._render(general)

    def visualize(self, general: GeneralData) -> Video:
        return self._render(general)

    def _render(self, general: GeneralData) -> Video:
        path = ensure_directory_exists(general.path / "videos") / f"{self.file_name}.mp4"

        frame_paths = sorted(
            general.path.glob(f"{self.image_name_prefix}*.png"),
            key = lambda x: x.name,
        )

        frame_paths = frame_paths[self.first_frame - 1:self.last_frame or int(1e9):self.frame_stride]

        if self.looping:
            final_frame_paths = chain(frame_paths, reversed(frame_paths[:-1]))
        else:
            final_frame_paths = frame_paths

        frame_list_path = path.with_suffix(".lst")
        save_text(frame_list_path, "".join(f"file '{x.resolve()}'\nduration 1\n" for x in final_frame_paths))
        run([
            "ffmpeg",
            "-y",
            "-r", str(self.fps),
            "-f", "concat",
            "-safe", "0",
            "-i", frame_list_path,
            "-framerate", str(self.fps),
            "-vf", ",".join([
                filter.print(self.fps)
                for filter in self.filters
                if filter.enabled
            ] or ["null"]),
            "-c:v", "libx264",
            "-crf", "14",
            "-preset", "slow" if self.archive_mode else "veryfast",
            "-tune", "film",
            "-pix_fmt", "yuv420p",
            path,
        ])
        frame_list_path.unlink()

        return Video(path = path, format = "mp4")
