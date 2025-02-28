from typing import Optional

from temporal.meta.serializable import Serializable, SerializableField as Field
from temporal.project import Project
from temporal.utils.image import NumpyImage
from temporal.vector import IntVector
from temporal.video_renderer import VideoRenderer


class Preset(Serializable):
    name: str = Field("")
    image: Optional[NumpyImage] = Field(None, variant = "image")
    image_size: IntVector = Field(factory = lambda: IntVector(512, 512))
    load_parameters: bool = Field(True)
    continue_from_last_frame: bool = Field(True)
    iter_count: int = Field(10)
    project: Project = Field(factory = Project)
    video_renderer: VideoRenderer = Field(factory = VideoRenderer)
    video_parallel_index: int = Field(1)
    measuring_parallel_index: int = Field(1)
