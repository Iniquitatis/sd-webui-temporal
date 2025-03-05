from typing import Iterator

from temporal.color import Color
from temporal.meta.configurable import ConfigurableParam as Param
from temporal.vector import FloatVector, IntVector
from temporal.video_filter import VideoFilter, make_filter as mf


class TextOverlayFilter(VideoFilter):
    name = "Text overlay"

    text: str = Param("Text", value = "{frame}", ui_type = "box")
    anchor: FloatVector = Param("Anchor", axes = ["X", "Y"], minimum = 0.0, maximum = 1.0, step = 0.01, factory = lambda: FloatVector(0.0, 0.0), ui_type = "slider")
    offset: IntVector = Param("Offset", axes = ["X", "Y"], step = 1, factory = lambda: IntVector(0, 0), ui_type = "box")
    font: str = Param("Font", value = "sans", ui_type = "box")
    font_size: int = Param("Font size", minimum = 1, maximum = 144, step = 1, value = 16, ui_type = "slider")
    text_color: Color = Param("Text color", channels = 4, factory = lambda: Color(1.0, 1.0, 1.0, 1.0))
    shadow_offset: IntVector = Param("Shadow offset", axes = ["X", "Y"], step = 1, factory = lambda: IntVector(1, 1), ui_type = "box")
    shadow_color: Color = Param("Shadow color", channels = 4, factory = lambda: Color(0.0, 0.0, 0.0, 1.0))

    def generate(self, fps: int) -> Iterator[str]:
        yield mf([], [], "drawtext",
            text = self.text.format(frame = f"%{{eif:t*{fps}+1:d:5}}"),
            x = f"(W-tw)*{self.anchor.x}+{self.offset.x}",
            y = f"(H-th)*{self.anchor.y}+{self.offset.y}",
            font = self.font,
            fontsize = self.font_size,
            fontcolor = self.text_color.to_hex(),
            shadowx = self.shadow_offset.x,
            shadowy = self.shadow_offset.y,
            shadowcolor = self.shadow_color.to_hex(),
        )
