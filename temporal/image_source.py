from typing import Literal, Optional

import numpy as np

from temporal.meta.serializable import Serializable, SerializableField as Field
from temporal.utils.image import NumpyImage, PILImage, pil_to_np
from temporal.video import Video


class ImageSource(Serializable):
    type: Literal["image", "initial_image", "video"] = Field("image")
    value: Optional[NumpyImage | Video] = Field(None)

    @property
    def image(self) -> Optional[NumpyImage]:
        return self.value if isinstance(self.value, np.ndarray) else None

    @property
    def video(self) -> Optional[Video]:
        return self.value if isinstance(self.value, Video) else None

    def get_image(self, initial_image: Optional[NumpyImage | PILImage], frame_index: int) -> Optional[NumpyImage]:
        if self.type == "image":
            return self.image
        elif self.type == "initial_image":
            return pil_to_np(initial_image) if isinstance(initial_image, PILImage) else initial_image
        elif self.type == "video" and self.video is not None and frame_index < self.video.get_frame_count():
            return self.video.get_frame(frame_index)
