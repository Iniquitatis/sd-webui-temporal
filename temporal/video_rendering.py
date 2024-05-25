from collections.abc import Sequence
from itertools import chain
from pathlib import Path
from subprocess import run
from types import SimpleNamespace

from temporal.thread_queue import ThreadQueue
from temporal.utils.fs import save_text
from temporal.video_filtering import build_filter

video_render_queue = ThreadQueue()

def enqueue_video_render(path: Path, frame_paths: Sequence[Path], ext_params: SimpleNamespace, is_final: bool) -> None:
    video_render_queue.enqueue(render_video, path, frame_paths, ext_params, is_final)

def render_video(path: Path, frame_paths: Sequence[Path], ext_params: SimpleNamespace, is_final: bool) -> None:
    if ext_params.video_looping:
        final_frame_paths = chain(frame_paths, reversed(frame_paths[:-1]))
    else:
        final_frame_paths = frame_paths

    frame_list_path = path.with_suffix(".lst")
    save_text(frame_list_path, "".join(f"file '{x.resolve()}'\nduration 1\n" for x in final_frame_paths))
    run([
        "ffmpeg",
        "-y",
        "-r", str(ext_params.video_fps),
        "-f", "concat",
        "-safe", "0",
        "-i", frame_list_path,
        "-framerate", str(ext_params.video_fps),
        "-vf", build_filter(ext_params) if is_final else "null",
        "-c:v", "libx264",
        "-crf", "14",
        "-preset", "slow" if is_final else "veryfast",
        "-tune", "film",
        "-pix_fmt", "yuv420p",
        path,
    ])
    frame_list_path.unlink()
