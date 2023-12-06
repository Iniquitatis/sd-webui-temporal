from pathlib import Path
from subprocess import run

from temporal.thread_queue import ThreadQueue
from temporal.video_filtering import build_filter

video_render_queue = ThreadQueue()

def start_video_render(ext_params, is_final):
    video_render_queue.enqueue(render_video, ext_params, is_final)

def render_video(ext_params, is_final):
    output_dir = Path(ext_params.output_dir)
    frame_dir = output_dir / ext_params.project_subdir
    frame_paths = sorted(frame_dir.glob("*.png"), key = lambda x: x.name)
    video_path = output_dir / f"{ext_params.project_subdir}-{'final' if is_final else 'draft'}.mp4"

    if ext_params.video_looping:
        frame_paths += reversed(frame_paths[:-1])

    run([
        "ffmpeg",
        "-y",
        "-r", str(ext_params.video_fps),
        "-f", "concat",
        "-protocol_whitelist", "fd,file",
        "-safe", "0",
        "-i", "-",
        "-framerate", str(ext_params.video_fps),
        "-vf", build_filter(ext_params) if is_final else "null",
        "-c:v", "libx264",
        "-crf", "14",
        "-preset", "slow" if is_final else "veryfast",
        "-tune", "film",
        "-pix_fmt", "yuv420p",
        video_path,
    ], input = "".join(f"file '{frame_path.resolve()}'\nduration 1\n" for frame_path in frame_paths).encode("utf-8"))
