from typing import Literal, Optional

from temporal.object import Field, Object
from temporal.utils.image import NumpyImage
from temporal.video import Video


class ImageSource(Object):
    type: Literal["image", "initial_image", "video"] = Field("image")
    image: Optional[NumpyImage] = Field(None)
    video: Optional[Video] = Field(None)

    def get_image(self, initial_image: Optional[NumpyImage], frame_index: int) -> Optional[NumpyImage]:
        if self.type == "image":
            return self.image
        elif self.type == "initial_image":
            return initial_image
        elif self.type == "video" and self.video is not None and frame_index < self.video.get_frame_count():
            return self.video.get_frame(frame_index)
