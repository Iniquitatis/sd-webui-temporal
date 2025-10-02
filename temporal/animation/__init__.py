from itertools import pairwise
from typing import Any, Literal, TypeVar, cast

from temporal.color import Color
from temporal.object import Field, Object
from temporal.utils.math import clamp, mirror, normalize, repeat


class Keyframe(Object):
    frame: int = Field(0)
    value: Any = Field(None)


InterpolationMode = Literal["linear", "smoothstep", "smootherstep", "step", "step_start", "step_end"]
BoundsMode = Literal["clamp", "repeat", "mirror"]


class Track(Object):
    key: str = Field("")
    interpolation: InterpolationMode = Field("linear")
    bounds: BoundsMode = Field("clamp")
    keyframes: list[Keyframe] = Field(list)

    @property
    def first_frame(self) -> int:
        return min(x.frame for x in self.keyframes)

    @property
    def last_frame(self) -> int:
        return max(x.frame for x in self.keyframes)

    @property
    def length(self) -> int:
        return self.last_frame - self.first_frame + 1

    def add_keyframe(self, frame: int, value: Any) -> None:
        self.keyframes.append(Keyframe(frame, value))
        self.keyframes.sort(key = lambda x: x.frame)

    def evaluate(self, frame: int) -> Any:
        if frame < self.first_frame or frame > self.last_frame:
            if self.bounds == "clamp":
                frame = clamp(frame, self.first_frame, self.last_frame)
            elif self.bounds == "repeat":
                frame = repeat(frame, self.first_frame, self.last_frame)
            elif self.bounds == "mirror":
                frame = mirror(frame, self.first_frame, self.last_frame)

        for keyframe_a, keyframe_b in pairwise(self.keyframes):
            frame_a = keyframe_a.frame
            frame_b = keyframe_b.frame

            if frame_a <= frame <= frame_b:
                value_a = keyframe_a.value
                value_b = keyframe_b.value

                return _interpolate(value_a, value_b, normalize(frame, frame_a, frame_b), self.interpolation)


class Animation(Object):
    tracks: list[Track] = Field(list)

    def evaluate(self, frame: int) -> dict[str, Any]:
        return {
            track.key: track.evaluate(frame)
            for track in self.tracks
        }


_T = TypeVar("_T")


def _interpolate(a: _T, b: _T, x: float, mode: InterpolationMode = "linear") -> _T:
    if mode == "linear":
        x = x
    elif mode == "smoothstep":
        x = x * x * (3.0 - 2.0 * x)
    elif mode == "smootherstep":
        x = x * x * x * (x * (6.0 * x - 15.0) + 10.0)
    elif mode == "step":
        x = 0.0 if x < 0.5 else 1.0
    elif mode == "step_start":
        x = 0.0 if (x + 0.5) <= 0.5 else 1.0
    elif mode == "step_end":
        x = 0.0 if (x - 0.5) < 0.5 else 1.0
    else:
        raise ValueError

    if isinstance(a, int) and isinstance(b, int):
        return cast(_T, round(float(a) * (1.0 - x) + float(b) * x))
    elif isinstance(a, float) and isinstance(b, float):
        return cast(_T, a * (1.0 - x) + b * x)
    elif isinstance(a, Color) and isinstance(b, Color):
        return cast(_T, Color(
            r = _interpolate(a.r, b.r, x),
            g = _interpolate(a.g, b.g, x),
            b = _interpolate(a.b, b.b, x),
            a = _interpolate(a.a, b.a, x),
        ))
    elif type(a) == type(b):
        return a if x < 0.5 else b
    else:
        raise ValueError
