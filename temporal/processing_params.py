from typing import Optional

from temporal.meta.serializable import Serializable, SerializableField as Field


class ProcessingParams(Serializable):
    model: str = Field("")
    vae: Optional[str] = Field("")
    clip_skip: int = Field(1)
    positive_prompt: str = Field("")
    negative_prompt: str = Field("")
    sampler: str = Field("")
    scheduler: str = Field("")
    steps: int = Field(20)
    cfg: float = Field(5.0)
    strength: float = Field(0.5)
    seed: int = Field(-1)
