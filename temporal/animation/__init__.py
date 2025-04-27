from itertools import pairwise
from typing import Generic, Literal, Optional, TypeVar

from temporal.color import Color
from temporal.object import Field, Object
from temporal.serialization import JSONValue, SerializationParams


Animatable = bool | int | float | str | Color

T = TypeVar("T", bound = Animatable)


class Keyframe(Object, Generic[T]):
    frame: int = Field(0)
    value: Optional[T] = Field(None)


InterpolationMode = Literal["linear", "smoothstep", "smootherstep", "step", "step_start", "step_end"]
BoundsMode = Literal["clamp", "repeat", "mirror"]


class Track(Object, Generic[T]):
    interpolation: InterpolationMode = Field("linear")
    bounds: BoundsMode = Field("clamp")
    keyframes: list[Keyframe[T]] = Field(list)

    @property
    def first_frame(self) -> int:
        return min(x.frame for x in self.keyframes)

    @property
    def last_frame(self) -> int:
        return max(x.frame for x in self.keyframes)

    @property
    def length(self) -> int:
        return self.last_frame - self.first_frame + 1

    def add_keyframe(self, frame: int, value: T) -> None:
        self.keyframes.append(Keyframe(frame, value))
        self.keyframes.sort(key = lambda x: x.frame)

    def evaluate(self, frame: int) -> Optional[T]:
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
    tracks: dict[str, Track[Animatable]] = Field(dict)

    # NOTE: Can't use Self, as its type relies on the external code
    @classmethod
    def from_json(cls, data: JSONValue, params: SerializationParams = SerializationParams()) -> "Animation":
        from temporal.animation.parsing import parse_animation

        if not isinstance(data, dict):
            raise ValueError

        if isinstance(code := data.get("code", None), str):
            return parse_animation(code)
        else:
            raise ValueError

    def to_json(self, params: SerializationParams = SerializationParams()) -> JSONValue:
        from temporal.animation.printing import print_animation

        return {"code": print_animation(self)}

    def get_track(self, property_name: str) -> Track[Animatable]:
        if (property := self.tracks.get(property_name, None)) is None:
            self.tracks[property_name] = property = Track()

        return property

    def evaluate(self, frame: int) -> dict[str, Optional[Animatable]]:
        return {
            property_name: track.evaluate(frame)
            for property_name, track in self.tracks.items()
        }


def _interpolate(a: T, b: T, x: float, mode: InterpolationMode = "linear") -> T:
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
