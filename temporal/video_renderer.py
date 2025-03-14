from collections.abc import Sequence
from itertools import chain
from pathlib import Path
from subprocess import run

from temporal.object import Field, Object
from temporal.thread_queue import ThreadQueue
from temporal.utils.fs import save_text
from temporal.video_filter import VideoFilter


# FIXME: Not sure if renderer should know about threading... or even exist at
# all
video_render_queue = ThreadQueue()


class VideoRenderer(Object):
    fps: int = Field(30)
    first_frame: int = Field(1)
    last_frame: int = Field(0)
    frame_stride: int = Field(1)
    looping: bool = Field(False)
    filters: list[VideoFilter] = Field(list)
    archive_mode: bool = Field(False)

    def render(self, path: Path, frame_paths: Sequence[Path], enqueue: bool = True) -> None:
        if enqueue:
            video_render_queue.enqueue(self._inner, path, frame_paths)
        else:
            self._inner(path, frame_paths)

    def _inner(self, path: Path, frame_paths: Sequence[Path]) -> None:
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
