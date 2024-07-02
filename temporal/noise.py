from temporal.meta.serializable import Serializable, SerializableField as Field


class Noise(Serializable):
    scale: int = Field(1)
    octaves: int = Field(1)
    lacunarity: float = Field(2.0)
    persistence: float = Field(0.5)
    seed: int = Field(0)
    use_global_seed: bool = Field(False)
