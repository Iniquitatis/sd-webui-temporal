from typing import Optional

from temporal.meta.serializable import Serializable, SerializableField as Field
from temporal.utils.collection import get_with_fallback
from temporal.utils.image import NumpyImage


class ProcessingParams(Serializable):
    model: str = Field("")
    vae: Optional[str] = Field("")
    clip_skip: int = Field(1)
    positive_prompts: list[str] = Field(factory = list)
    negative_prompts: list[str] = Field(factory = list)
    width: int = Field(512)
    height: int = Field(512)
    sampler: str = Field("")
    scheduler: str = Field("")
    steps: int = Field(20)
    cfg: float = Field(5.0)
    strength: float = Field(0.5)
    seeds: list[int] = Field(factory = list)

    @property
    def positive_prompt(self) -> str:
        return get_with_fallback(self.positive_prompts, 0, "")

    @positive_prompt.setter
    def positive_prompt(self, value: str) -> None:
        self.positive_prompts = [value]

    @property
    def negative_prompt(self) -> str:
        return get_with_fallback(self.negative_prompts, 0, "")

    @negative_prompt.setter
    def negative_prompt(self, value: str) -> None:
        self.negative_prompts = [value]

    @property
    def seed(self) -> int:
        return get_with_fallback(self.seeds, 0, 0)

    @seed.setter
    def seed(self, value: int) -> None:
        self.seeds = [value]


class TextToImageParams(ProcessingParams):
    pass


class ImageToImageParams(ProcessingParams):
    images: list[NumpyImage] = Field(factory = list)

    @property
    def image(self) -> Optional[NumpyImage]:
        return get_with_fallback(self.images, 0)

    @image.setter
    def image(self, value: NumpyImage) -> None:
        self.images = [value]
