from typing import Any

from temporal.animation import Animation
from temporal.color import Color


def print_animation(animation: Animation, indentation: int = 4) -> str:
    lines = []

    spaces = " " * indentation

    for property_name, track in animation.tracks.items():
        lines.append(f"track {property_name}:")
        lines.append(f"{spaces}interpolation: {track.interpolation}")
        lines.append(f"{spaces}bounds: {track.bounds}")
        lines.append(f"")

        digits = len(str(track.last_frame))

        for keyframe in track.keyframes:
            lines.append(f"{spaces}frame {keyframe.frame:0{digits}d}: {_format_value(keyframe.value)}")

        lines.append(f"")

    return "\n".join(lines)


def _format_value(value: Any) -> str:
    if isinstance(value, str):
        return f"\"{value}\""
    elif isinstance(value, Color):
        return f"Color({value.r}, {value.g}, {value.b}, {value.a})"
    else:
        return f"{value}"
