from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

from temporal.meta.serializable import Serializable, field
from temporal.utils.image import PILImage


class OutputParams(Serializable):
    output_dir: Path = field(Path("outputs/temporal"), saved = False)
    project_subdir: str = field("untitled", saved = False)
    frame_count: int = field(1, saved = False)
    save_every_nth_frame: int = field(1)
    archive_mode: bool = field(False)


class InitialNoiseParams(Serializable):
    factor: float = field(0.0)
    scale: int = field(1)
    octaves: int = field(1)
    lacunarity: float = field(2.0)
    persistence: float = field(0.5)


class ProcessingParams(Serializable):
    use_sd: bool = field(True)
    show_only_finalized_frames: bool = field(False, saved = False)


class MultisamplingParams(Serializable):
    samples: int = field(1)
    batch_size: int = field(1)
    trimming: float = field(0.0)
    easing: float = field(0.0)
    preference: float = field(0.0)


class DetailingParams(Serializable):
    enabled: bool = field(False)
    scale: float = field(1.0)
    scale_buffer: bool = field(False)
    sampler: str = field("Euler a")
    steps: int = field(15)
    denoising_strength: float = field(0.2)


class FrameMergingParams(Serializable):
    frames: int = field(1)
    trimming: float = field(0.0)
    easing: float = field(0.0)
    preference: float = field(0.0)


class ProjectParams(Serializable):
    load_parameters: bool = field(False)
    continue_from_last_frame: bool = field(False)


class MaskParams(Serializable):
    image: Optional[PILImage] = field(None)
    normalized: bool = field(False)
    inverted: bool = field(False)
    blurring: float = field(0.0)


class ImageFilterParams(Serializable):
    enabled: bool = field(False)
    amount: float = field(1.0)
    amount_relative: bool = field(False)
    blend_mode: str = field("normal")
    params: SimpleNamespace = field(factory = SimpleNamespace)
    mask: MaskParams = field(factory = MaskParams)


class ImageFilteringParams(Serializable):
    filter_order: list[str] = field(factory = list)
    filter_data: defaultdict[str, ImageFilterParams] = field(factory = lambda: defaultdict(ImageFilterParams))


class VideoFilterParams(Serializable):
    enabled: bool = field(False)
    params: SimpleNamespace = field(factory = SimpleNamespace)


class VideoRenderingParams(Serializable):
    fps: int = field(30)
    looping: bool = field(False)
    filter_order: list[str] = field(factory = list)
    filter_data: defaultdict[str, VideoFilterParams] = field(factory = lambda: defaultdict(VideoFilterParams))


class MeasuringParams(Serializable):
    enabled: bool = field(False)
    plot_every_nth_frame: int = field(1)


class ExtensionData(Serializable):
    mode: str = field("sequence", saved = False)
    output: OutputParams = field(factory = OutputParams)
    initial_noise: InitialNoiseParams = field(factory = InitialNoiseParams)
    processing: ProcessingParams = field(factory = ProcessingParams)
    multisampling: MultisamplingParams = field(factory = MultisamplingParams)
    detailing: DetailingParams = field(factory = DetailingParams)
    frame_merging: FrameMergingParams = field(factory = FrameMergingParams)
    project: ProjectParams = field(factory = ProjectParams, saved = False)
    filtering: ImageFilteringParams = field(factory = ImageFilteringParams)
    video: VideoRenderingParams = field(factory = VideoRenderingParams, saved = False)
    render_draft_on_finish: bool = field(False, saved = False)
    render_final_on_finish: bool = field(False, saved = False)
    measuring: MeasuringParams = field(factory = MeasuringParams, saved = False)
