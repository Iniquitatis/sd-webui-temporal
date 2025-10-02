from typing import Iterator

from modules.color import Color
from modules.object import Field
from modules.vector import FloatVector, IntVector
from modules.video_filter import VideoFilter, make_filter as mf


class TextOverlayFilter(VideoFilter):
    name = "Text overlay"

    text: str = Field("{frame}", name = "Text", display = "box")
    anchor: FloatVector = Field(lambda: FloatVector(0.0, 0.0), name = "Anchor", axes = ["X", "Y"], minimum = 0.0, maximum = 1.0, step = 0.01, display = "slider")
    offset: IntVector = Field(lambda: IntVector(0, 0), name = "Offset", axes = ["X", "Y"], step = 1, suffix = " px", display = "box")
    font: str = Field("sans", name = "Font", display = "box")
    font_size: int = Field(16, name = "Font size", minimum = 1, maximum = 144, step = 1, display = "slider")
    text_color: Color = Field(lambda: Color(1.0, 1.0, 1.0, 1.0), name = "Text color", channels = 4)
    shadow_offset: IntVector = Field(lambda: IntVector(1, 1), name = "Shadow offset", axes = ["X", "Y"], step = 1, suffix = " px", display = "box")
    shadow_color: Color = Field(lambda: Color(0.0, 0.0, 0.0, 1.0), name = "Shadow color", channels = 4)

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
