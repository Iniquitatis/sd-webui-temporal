from temporal.meta.serializable import Serializable, SerializableField as Field
from temporal.noise import Noise


class InitialNoiseParams(Serializable):
    factor: float = Field(0.0)
    noise: Noise = Field(factory = Noise)
