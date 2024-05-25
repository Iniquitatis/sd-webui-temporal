from temporal.compat import VERSION, upgrade_project
from temporal.fs import clear_directory, is_directory_empty, remove_entry, save_text
from temporal.image_utils import load_image

FRAME_NAME_FORMAT = "{index:05d}"
FRAME_EXTENSION = "png"
VIDEO_NAME_FORMAT = "{name}-{suffix}"
VIDEO_EXTENSION = "mp4"
VIDEO_DRAFT_SUFFIX = "draft"
VIDEO_FINAL_SUFFIX = "final"

def make_frame_name(index):
    return FRAME_NAME_FORMAT.format(index = index)

def make_frame_file_name(index):
    return f"{make_frame_name(index)}.{FRAME_EXTENSION}"

def make_video_name(name, is_final):
    return VIDEO_NAME_FORMAT.format(
        name = name,
        suffix = VIDEO_FINAL_SUFFIX if is_final else VIDEO_DRAFT_SUFFIX,
    )

def make_video_file_name(name, is_final):
    return f"{make_video_name(name, is_final)}.{VIDEO_EXTENSION}"

class Project:
    def __init__(self, path):
        self.path = path

    @property
    def name(self):
        return self.path.name

    @property
    def data_path(self):
        return self.path / "project"

    @property
    def session_path(self):
        return self.data_path / "session"

    @property
    def buffer_path(self):
        return self.data_path / "buffer"

    @property
    def metrics_path(self):
        return self.data_path / "metrics"

    def load(self):
        if is_directory_empty(self.path):
            return

        upgrade_project(self.path)

    def save(self):
        save_text(self.data_path / "version.txt", str(VERSION))

    def get_first_frame_index(self):
        return min((_parse_frame_index(x) for x in self._iterate_frame_paths()), default = 0)

    def get_last_frame_index(self):
        return max((_parse_frame_index(x) for x in self._iterate_frame_paths()), default = 0)

    def get_actual_frame_count(self):
        return sum(1 for _ in self._iterate_frame_paths())

    def list_all_frame_paths(self):
        return sorted(self._iterate_frame_paths(), key = lambda x: x.name)

    def load_frame(self, index):
        if not (path := (self.path / make_frame_file_name(index))).is_file():
            return None

        return load_image(path)

    def delete_all_frames(self):
        clear_directory(self.path, f"*.{FRAME_EXTENSION}")

    def delete_intermediate_frames(self):
        kept_paths = set()
        min_index = int(1e9)
        max_index = 0

        for image_path in self._iterate_frame_paths():
            if frame_index := _parse_frame_index(image_path):
                min_index = min(min_index, frame_index)
                max_index = max(max_index, frame_index)
            else:
                kept_paths.add(image_path)

        kept_paths.add(self.path / make_frame_file_name(min_index))
        kept_paths.add(self.path / make_frame_file_name(max_index))

        for image_path in self._iterate_frame_paths():
            if image_path not in kept_paths:
                remove_entry(image_path)

    def delete_session_data(self):
        remove_entry(self.buffer_path)
        remove_entry(self.metrics_path)

    def _iterate_frame_paths(self):
        return self.path.glob(f"*.{FRAME_EXTENSION}")

def _parse_frame_index(image_path):
    if image_path.is_file():
        try:
            return int(image_path.stem)
        except:
            print(f"WARNING: {image_path.stem} doesn't match the frame name format")

    return 0
