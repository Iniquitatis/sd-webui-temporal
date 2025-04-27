from typing import Optional

from temporal.object import Field, Object
from temporal.utils.image import NumpyImage
from temporal.video import Video


class ImageSource(Object):
    type: str = Field("image", name = "Type", choices = {
        "image": "Image",
        "initial_image": "Initial image",
        "video": "Video",
    }, display = "radio")
    image: Optional[NumpyImage] = Field(None, name = "Image", dependencies = {"type": "image"})
    video: Optional[Video] = Field(None, name = "Video", dependencies = {"type": "video"})

    def get_image(self, initial_image: Optional[NumpyImage], frame_index: int) -> Optional[NumpyImage]:
        if self.type == "image":
            return self.image
        elif self.type == "initial_image":
            return initial_image
        elif self.type == "video":
            if self.video is not None and frame_index < self.video.get_frame_count():
                return self.video.get_frame(frame_index)
            else:
                return None
        else:
            raise ValueError
