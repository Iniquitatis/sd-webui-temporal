from pathlib import Path
from subprocess import run

def render_video(uv, is_final):
    output_dir = Path(uv.output_dir)
    frame_dir = output_dir / uv.project_subdir
    frame_paths = sorted(frame_dir.glob("*.png"), key = lambda x: x.name)
    video_path = output_dir / f"{uv.project_subdir}-{'final' if is_final else 'draft'}.mp4"

    if uv.video_looping:
        frame_paths += reversed(frame_paths[:-1])

    filters = []

    if is_final:
        if uv.video_deflickering_enabled:
            filters.append(f"deflicker='size={min(uv.video_deflickering_frames, len(frame_paths))}:mode=am'")

        if uv.video_interpolation_enabled:
            filters.append(f"minterpolate='fps={uv.video_interpolation_fps * (uv.video_interpolation_mb_subframes + 1)}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1:scd=none'")

            if uv.video_interpolation_mb_subframes > 0:
                filters.append(f"tmix='frames={uv.video_interpolation_mb_subframes + 1}'")
                filters.append(f"fps='{uv.video_interpolation_fps}'")

        if uv.video_temporal_blurring_enabled:
            weights = [((x + 1) / (uv.video_temporal_blurring_radius + 1)) ** uv.video_temporal_blurring_easing for x in range(uv.video_temporal_blurring_radius + 1)]
            weights += reversed(weights[:-1])
            weights = [f"{x:.18f}" for x in weights]
            filters.append(f"tmix='frames={len(weights)}:weights={' '.join(weights)}'")

        if uv.video_scaling_enabled:
            filters.append(f"scale='{uv.video_scaling_width}x{uv.video_scaling_height}:flags=lanczos'")

    if uv.video_frame_num_overlay_enabled:
        filters.append(f"drawtext='text=%{{eif\\:n*{uv.video_fps / uv.video_interpolation_fps if is_final and uv.video_interpolation_enabled else 1.0:.18f}+1\\:d\\:5}}:x=5:y=5:fontsize={uv.video_frame_num_overlay_font_size}:fontcolor={uv.video_frame_num_overlay_text_color}{int(uv.video_frame_num_overlay_text_alpha * 255.0):02x}:shadowx=1:shadowy=1:shadowcolor={uv.video_frame_num_overlay_shadow_color}{int(uv.video_frame_num_overlay_shadow_alpha * 255.0):02x}'")

    run([
        "ffmpeg",
        "-y",
        "-r", str(uv.video_fps),
        "-f", "concat",
        "-protocol_whitelist", "fd,file",
        "-safe", "0",
        "-i", "-",
        "-framerate", str(uv.video_fps),
        "-vf", ",".join(filters) if len(filters) > 0 else "null",
        "-c:v", "libx264",
        "-crf", "14",
        "-preset", "slow" if is_final else "veryfast",
        "-tune", "film",
        "-pix_fmt", "yuv420p",
        video_path,
    ], input = "".join(f"file '{frame_path.resolve()}'\nduration 1\n" for frame_path in frame_paths).encode("utf-8"))
