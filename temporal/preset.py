from typing import Optional

from temporal.meta.serializable import Serializable, SerializableField as Field
from temporal.project import Project
from temporal.utils.image import NumpyImage
from temporal.video_renderer import VideoRenderer


class Preset(Serializable):
    # FIXME: Should only hold the data dictionary here
    image: Optional[NumpyImage] = Field(None, variant = "image")
    # FIXME: Doesn't play well with the new Session thing (handle Any in
    # from_/to_json)
    # load_parameters: bool = Field(True)
    # continue_from_last_frame: bool = Field(True)
    # iter_count: int = Field(10)
    project: Project = Field(factory = Project)
    video_renderer: VideoRenderer = Field(factory = VideoRenderer)
