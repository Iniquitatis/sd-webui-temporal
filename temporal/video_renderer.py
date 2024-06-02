from collections.abc import Sequence
from itertools import chain
from pathlib import Path
from subprocess import run

from temporal.meta.serializable import Serializable, field
from temporal.thread_queue import ThreadQueue
from temporal.utils.collection import reorder_dict
from temporal.utils.fs import save_text
from temporal.video_filters import VIDEO_FILTERS, VideoFilter


video_render_queue = ThreadQueue()


class VideoRenderer(Serializable):
    fps: int = field(30)
    looping: bool = field(False)
    filter_order: list[str] = field(factory = list)
    filters: dict[str, VideoFilter] = field(factory = lambda: {id: cls() for id, cls in VIDEO_FILTERS.items()})

    def enqueue_video_render(self, path: Path, frame_paths: Sequence[Path], is_final: bool) -> None:
        video_render_queue.enqueue(self._render_video, path, frame_paths, is_final)

    def _render_video(self, path: Path, frame_paths: Sequence[Path], is_final: bool) -> None:
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
            "-vf", self._build_filter() if is_final else "null",
            "-c:v", "libx264",
            "-crf", "14",
            "-preset", "slow" if is_final else "veryfast",
            "-tune", "film",
            "-pix_fmt", "yuv420p",
            path,
        ])
        frame_list_path.unlink()

    def _build_filter(self):
        return ",".join([
            filter.print(self.fps)
            for filter in reorder_dict(self.filters, self.filter_order).values()
            if filter.enabled
        ] or ["null"])
