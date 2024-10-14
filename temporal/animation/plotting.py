from io import BytesIO

import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.ticker import MaxNLocator

from temporal.animation import Animation, Track
from temporal.color import Color
from temporal.utils.image import PILImage


def plot_animation(animation: Animation) -> list[PILImage]:
    return [
        _plot_track(k, v, 1, v.last_frame * 2)
        for k, v in animation.tracks.items()
    ]


def _plot_track(property_name: str, track: Track, first_frame: int = 1, last_frame: int = 60, bounds: int = 2) -> PILImage:
    keyframe_frames = []
    keyframe_values = []
    evaluated_frames = []
    evaluated_values = []

    for keyframe in track.keyframes:
        keyframe_frames.append(keyframe.frame)
        keyframe_values.append(keyframe.value)

    for i in range(first_frame, last_frame + 1):
        evaluated_frames.append(i)
        evaluated_values.append(track.evaluate(i))

    plt.title(property_name)
    plt.xlabel("Frame")
    plt.xticks(evaluated_frames)
    plt.xlim(first_frame - bounds, last_frame + bounds)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer = True))
    plt.axvspan(first_frame - bounds, first_frame, facecolor = (0.0, 0.0, 0.0, 0.5))
    plt.axvspan(last_frame, last_frame + bounds, facecolor = (0.0, 0.0, 0.0, 0.5))
    plt.ylabel("Value")
    plt.grid()

    _draw_markers(keyframe_frames)

    if isinstance(keyframe_values[0], bool):
        _draw_keyframes(keyframe_frames, [float(x) for x in keyframe_values])
        _draw_curve(evaluated_frames, [float(x) for x in evaluated_values])

    elif isinstance(keyframe_values[0], float):
        _draw_keyframes(keyframe_frames, keyframe_values)
        _draw_curve(evaluated_frames, evaluated_values)

    elif isinstance(keyframe_values[0], str):
        _draw_keyframes(keyframe_frames, _make_text_curve(keyframe_values, _extract_text_indices(keyframe_values)))
        _draw_curve(evaluated_frames, _make_text_curve(evaluated_values, _extract_text_indices(keyframe_values)))
        _draw_annotations(keyframe_frames, _make_text_curve(keyframe_values, _extract_text_indices(keyframe_values)), keyframe_values, "bold")
        _draw_annotations(evaluated_frames, _make_text_curve(evaluated_values, _extract_text_indices(keyframe_values)), evaluated_values, "normal", Color(0.25, 0.25, 0.25), "top")

    elif isinstance(keyframe_values[0], Color):
        _draw_keyframes(keyframe_frames, [x.r for x in keyframe_values], Color(1.0, 0.4, 0.4))
        _draw_keyframes(keyframe_frames, [x.g for x in keyframe_values], Color(0.4, 1.0, 0.4))
        _draw_keyframes(keyframe_frames, [x.b for x in keyframe_values], Color(0.4, 0.4, 1.0))
        _draw_keyframes(keyframe_frames, [x.a for x in keyframe_values], Color(0.5, 0.5, 0.5))
        _draw_curve(evaluated_frames, [x.r for x in evaluated_values], Color(1.0, 0.5, 0.5, 0.75))
        _draw_curve(evaluated_frames, [x.g for x in evaluated_values], Color(0.5, 1.0, 0.5, 0.75))
        _draw_curve(evaluated_frames, [x.b for x in evaluated_values], Color(0.5, 0.5, 1.0, 0.75))
        _draw_curve(evaluated_frames, [x.a for x in evaluated_values], Color(0.6, 0.6, 0.6, 0.75))

    plt.legend(loc = "upper right")

    buffer = BytesIO()
    plt.savefig(buffer, format = "png")
    buffer.seek(0)

    im = Image.open(buffer)
    im.load()

    plt.close()

    return im


def _extract_text_indices(texts: list[str]) -> dict[str, int]:
    result = {}

    for i, text in enumerate(texts):
        if text not in result:
            result[text] = i + 1

    return result


def _make_text_curve(texts: list[str], indices: dict[str, int]) -> list[float]:
    return [float(indices[x]) for x in texts]


def _draw_markers(frames: list[int]) -> None:
    for x in frames:
        plt.axvline(x, linestyle = "--", linewidth = 1.5)


def _draw_keyframes(frames: list[int], values: list[float], color: Color = Color(1.0, 0.5, 0.5)) -> None:
    plt.plot(
        frames,
        values,
        label = "Keyframes",
        color = (color.r, color.g, color.b, color.a),
        linewidth = 3.0,
        marker = "D",
        markeredgecolor = (color.r, color.g, color.b, color.a),
        markeredgewidth = 2.0,
        markerfacecolor = (0.0, 0.0, 0.0, 0.0),
        markersize = 8,
    )


def _draw_curve(frames: list[int], values: list[float], color: Color = Color(0.5, 0.5, 1.0)) -> None:
    plt.axhline(min(values), linestyle = ":", linewidth = 1.0)
    plt.axhline(max(values), linestyle = ":", linewidth = 1.0)
    plt.plot(
        frames,
        values,
        label = "Evaluated",
        color = (color.r, color.g, color.b, color.a * 0.75),
        linestyle = "solid",
        linewidth = 1.5,
        marker = "o",
        markeredgecolor = (color.r, color.g, color.b, color.a),
        markeredgewidth = 1.5,
        markerfacecolor = (1.0, 1.0, 0.5, color.a),
        markersize = 5,
    )


def _draw_annotations(frames: list[int], values: list[float], texts: list[str], weight: str = "normal", color: Color = Color(0.0, 0.0, 0.0), alignment: str = "bottom") -> None:
    last_text = ""

    for frame, value, text in zip(frames, values, texts):
        if text == last_text:
            continue

        plt.annotate(
            text,
            (frame, value),
            xytext = (5, 5 if alignment == "bottom" else -5),
            textcoords = "offset points",
            horizontalalignment = "left",
            verticalalignment = alignment,
            backgroundcolor = (1.0, 1.0, 1.0, 0.35),
            color = (color.r, color.g, color.b, color.a),
            fontweight = weight,
        )

        last_text = text
