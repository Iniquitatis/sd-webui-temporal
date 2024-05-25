from itertools import chain
from subprocess import run

from temporal.thread_queue import ThreadQueue
from temporal.utils.fs import save_text
from temporal.video_filtering import build_filter

video_render_queue = ThreadQueue()

def enqueue_video_render(path, frame_paths, ext_params, is_final):
    video_render_queue.enqueue(render_video, path, frame_paths, ext_params, is_final)

def render_video(path, frame_paths, ext_params, is_final):
    if ext_params.video_looping:
        frame_paths = chain(frame_paths, reversed(frame_paths[:-1]))

    frame_list_path = path.with_suffix(".lst")
    save_text(frame_list_path, "".join(f"file '{x.resolve()}'\nduration 1\n" for x in frame_paths))
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
