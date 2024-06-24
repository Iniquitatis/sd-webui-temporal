from temporal.fs_store import FSStore
from temporal.preset import Preset


class PresetStore(FSStore[Preset]):
    def __create_entry__(self, name: str) -> Preset:
        return Preset()
