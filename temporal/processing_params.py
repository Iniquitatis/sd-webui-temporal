from random import randint
from typing import Any, Optional

from temporal.meta.serializable import Serializable, SerializableField as Field


class ProcessingParams(Serializable):
    model: str = Field("")
    vae: Optional[str] = Field(None)
    clip_skip: int = Field(1)
    positive_prompt: str = Field("")
    negative_prompt: str = Field("")
    sampler: str = Field("")
    scheduler: str = Field("")
    steps: int = Field(20)
    cfg: float = Field(5.0)
    strength: float = Field(0.5)
    seed: int = Field(-1)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        if self.seed == -1:
            self.seed = randint(0, 0x7fffffff)
