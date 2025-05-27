from itertools import pairwise
from typing import Any, Literal, Optional, TypeVar

from temporal.color import Color
from temporal.object import Field, Object


Animatable = bool | int | float | str | Color


class Keyframe(Object):
    frame: int = Field(0)
    value: Optional[Any] = Field(None)


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

    def evaluate(self, frame: int) -> Optional[Any]:
        if frame < self.first_frame:
            if self.bounds == "clamp":
                frame = self.first_frame
            elif self.bounds == "repeat":
                frame -= self.last_frame
                frame %= self.length
                frame += self.last_frame
            elif self.bounds == "mirror":
                frame -= self.last_frame
                frame = (self.length - 1) - abs(frame % ((self.length - 1) * 2) - (self.length - 1))
                frame += self.last_frame

        if frame > self.last_frame:
            if self.bounds == "clamp":
                frame = self.last_frame
            elif self.bounds == "repeat":
                frame -= self.first_frame
                frame %= self.length
                frame += self.first_frame
            elif self.bounds == "mirror":
                frame -= self.first_frame
                frame = (self.length - 1) - abs(frame % ((self.length - 1) * 2) - (self.length - 1))
                frame += self.first_frame

        for keyframe_a, keyframe_b in pairwise(self.keyframes):
            frame_a = keyframe_a.frame
            frame_b = keyframe_b.frame

            if frame_a <= frame <= frame_b:
                value_a = keyframe_a.value
                value_b = keyframe_b.value

                if value_b is None:
                    value_b = value_a

                return _interpolate(value_a, value_b, (frame - frame_a) / (frame_b - frame_a), self.interpolation)


class Animation(Object):
    tracks: list[Track] = Field(list)

    def evaluate(self, frame: int) -> dict[str, Optional[Animatable]]:
        return {
            track.key: track.evaluate(frame)
            for track in self.tracks
        }


_T = TypeVar("_T", bound = Animatable)


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

    if isinstance(a, bool) and isinstance(b, bool):
        return a if x < 0.5 else b
    elif isinstance(a, int) and isinstance(b, int):
        return round(float(a) * (1.0 - x) + float(b) * x)
    elif isinstance(a, float) and isinstance(b, float):
        return a * (1.0 - x) + b * x
    elif isinstance(a, str) and isinstance(b, str):
        return a if x < 0.5 else b
    elif isinstance(a, Color) and isinstance(b, Color):
        return Color(
            r = _interpolate(a.r, b.r, x),
            g = _interpolate(a.g, b.g, x),
            b = _interpolate(a.b, b.b, x),
            a = _interpolate(a.a, b.a, x),
        )
    else:
        raise ValueError
