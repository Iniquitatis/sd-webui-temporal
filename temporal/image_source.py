from typing import Any, Optional, Self

from modules.object import Field, Object
from modules.utils.image import NumpyImage, ensure_image_dims
from modules.video import Video


class ImageSource(Object):
    type: str = Field("image", name = "Type", choices = {
        "image": "Image",
        "video": "Video",
        "initial_image": "Initial image",
    }, display = "radio")
    image: Optional[NumpyImage] = Field(None, name = "Image", dependencies = {"type": "image"})
    video: Optional[Video] = Field(None, name = "Video", dependencies = {"type": "video"})
    channels: int = Field(4, flags = {"runtime"})

    def validate(self, criteria: dict[str, Any]) -> Self:
        super().validate(criteria)

        if (channels := criteria.get("channels")) is not None:
            self.channels = channels

        return self

    def get_image(self, initial_image: Optional[NumpyImage], frame_index: int) -> Optional[NumpyImage]:
        result = None

        if self.type == "image" and self.image is not None:
            result = self.image
        elif self.type == "video" and self.video is not None and frame_index < self.video.get_frame_count():
            result = self.video.get_frame(frame_index)
        elif self.type == "initial_image":
            result = initial_image

        if result is not None:
            return ensure_image_dims(result, channels = self.channels)
