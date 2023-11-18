from pathlib import Path
from subprocess import run

from temporal.thread_queue import ThreadQueue

video_render_queue = ThreadQueue()

def start_video_render(ext_params, is_final, metadata = ""):
    video_render_queue.enqueue(render_video, ext_params, is_final, metadata)

def render_video(ext_params, is_final, metadata = ""):
    output_dir = Path(ext_params.output_dir)
    frame_dir = output_dir / ext_params.project_subdir
    frame_paths = sorted(frame_dir.glob("*.png"), key = lambda x: x.name)
    video_path = output_dir / f"{ext_params.project_subdir}-{'final' if is_final else 'draft'}.mp4"

    if ext_params.video_looping:
        frame_paths += reversed(frame_paths[:-1])

    filters = []

    if is_final:
        if ext_params.video_deflickering_enabled:
            filters.append(f"deflicker='size={min(ext_params.video_deflickering_frames, len(frame_paths))}:mode=am'")

        if ext_params.video_interpolation_enabled:
            filters.append(f"minterpolate='fps={ext_params.video_interpolation_fps * (ext_params.video_interpolation_mb_subframes + 1)}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1:scd=none'")

            if ext_params.video_interpolation_mb_subframes > 0:
                filters.append(f"tmix='frames={ext_params.video_interpolation_mb_subframes + 1}'")
                filters.append(f"fps='{ext_params.video_interpolation_fps}'")

        if ext_params.video_temporal_blurring_enabled:
            weights = [((x + 1) / (ext_params.video_temporal_blurring_radius + 1)) ** ext_params.video_temporal_blurring_easing for x in range(ext_params.video_temporal_blurring_radius + 1)]
            weights += reversed(weights[:-1])
            weights = [f"{x:.18f}" for x in weights]
            filters.append(f"tmix='frames={len(weights)}:weights={' '.join(weights)}'")

        if ext_params.video_scaling_enabled:
            filters.append(f"scale='{ext_params.video_scaling_width}x{ext_params.video_scaling_height}:flags=lanczos'")

    if ext_params.video_frame_num_overlay_enabled:
        filters.append(f"drawtext='text=%{{eif\\:n*{ext_params.video_fps / ext_params.video_interpolation_fps if is_final and ext_params.video_interpolation_enabled else 1.0:.18f}+1\\:d\\:5}}:x=5:y=5:fontsize={ext_params.video_frame_num_overlay_font_size}:fontcolor={ext_params.video_frame_num_overlay_text_color}{int(ext_params.video_frame_num_overlay_text_alpha * 255.0):02x}:shadowx=1:shadowy=1:shadowcolor={ext_params.video_frame_num_overlay_shadow_color}{int(ext_params.video_frame_num_overlay_shadow_alpha * 255.0):02x}'")

    run([
        "ffmpeg",
        "-y",
        "-r", str(ext_params.video_fps),
        "-f", "concat",
        "-protocol_whitelist", "fd,file",
        "-safe", "0",
        "-i", "-",
        "-framerate", str(ext_params.video_fps),
        "-vf", ",".join(filters) if len(filters) > 0 else "null",
        "-c:v", "libx264",
        "-crf", "14",
        "-preset", "slow" if is_final else "veryfast",
        "-tune", "film",
        "-pix_fmt", "yuv420p",
        "-metadata", f"parameters={metadata}",
        "-movflags",
        "+use_metadata_tags",
        video_path,
    ], input = "".join(f"file '{frame_path.resolve()}'\nduration 1\n" for frame_path in frame_paths).encode("utf-8"))
