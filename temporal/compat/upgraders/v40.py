import xml.etree.ElementTree as ET
from pathlib import Path

from modules.compat.upgrader import Upgrader


class _(Upgrader):
    version = 40

    def upgrade(self, path: Path) -> bool:
        data_path = path / "project" / "data.xml"

        if not data_path.exists():
            return False

        tree = ET.ElementTree(file = data_path)

        if tree.findtext("*[@key='version']", "0") != str(self.previous_version):
            return False

        for src_key, dst_key, dst_type in [
            ("blurring", "blurring", "temporal.pipeline_modules.filtering.blurring.BlurringFilter"),
            ("color_balancing", "color_balancing", "temporal.pipeline_modules.filtering.color_balancing.ColorBalancingFilter"),
            ("color_correction", "color_correction", "temporal.pipeline_modules.filtering.color_correction.ColorCorrectionFilter"),
            ("custom_code", "custom_code", "temporal.pipeline_modules.filtering.custom_code.CustomCodeFilter"),
            ("median", "median", "temporal.pipeline_modules.filtering.median.MedianFilter"),
            ("morphology", "morphology", "temporal.pipeline_modules.filtering.morphology.MorphologyFilter"),
            ("noise_compression", "noise_compression", "temporal.pipeline_modules.filtering.noise_compression.NoiseCompressionFilter"),
            ("palettization", "palettization", "temporal.pipeline_modules.filtering.palettization.PalettizationFilter"),
            ("pixelization", "pixelization", "temporal.pipeline_modules.filtering.pixelization.PixelizationFilter"),
            ("sharpening", "sharpening", "temporal.pipeline_modules.filtering.sharpening.SharpeningFilter"),
            ("symmetry", "symmetry", "temporal.pipeline_modules.filtering.symmetry.SymmetryFilter"),
            ("transformation", "transformation", "temporal.pipeline_modules.filtering.transformation.TransformationFilter"),

            ("color_level_mean_measuring", "color_level_mean_measuring", "temporal.pipeline_modules.measuring.color_level_mean.ColorLevelMeanMeasuringModule"),
            ("color_level_sigma_measuring", "color_level_sigma_measuring", "temporal.pipeline_modules.measuring.color_level_sigma.ColorLevelSigmaMeasuringModule"),
            ("luminance_mean_measuring", "luminance_mean_measuring", "temporal.pipeline_modules.measuring.luminance_mean.LuminanceMeanMeasuringModule"),
            ("luminance_sigma_measuring", "luminance_sigma_measuring", "temporal.pipeline_modules.measuring.luminance_sigma.LuminanceSigmaMeasuringModule"),
            ("noise_sigma_measuring", "noise_sigma_measuring", "temporal.pipeline_modules.measuring.noise_sigma.NoiseSigmaMeasuringModule"),

            ("detailing", "detailing", "temporal.pipeline_modules.neural.detailing.DetailingModule"),
            ("processing", "processing", "temporal.pipeline_modules.neural.processing.ProcessingModule"),

            ("color_overlay", "color_painting", "temporal.pipeline_modules.painting.color.ColorPaintingModule"),
            ("gradient_overlay", "gradient_painting", "temporal.pipeline_modules.painting.gradient.GradientPaintingModule"),
            ("image_overlay", "image_painting", "temporal.pipeline_modules.painting.image.ImagePaintingModule"),
            ("noise_overlay", "noise_painting", "temporal.pipeline_modules.painting.noise.NoisePaintingModule"),
            ("pattern_overlay", "pattern_painting", "temporal.pipeline_modules.painting.pattern.PatternPaintingModule"),

            ("averaging", "averaging", "temporal.pipeline_modules.temporal.averaging.AveragingModule"),
            ("interpolation", "interpolation", "temporal.pipeline_modules.temporal.interpolation.InterpolationModule"),
            ("limiting", "limiting", "temporal.pipeline_modules.temporal.limiting.LimitingModule"),
            ("random_sampling", "random_sampling", "temporal.pipeline_modules.temporal.random_sampling.RandomSamplingModule"),

            ("saving", "saving", "temporal.pipeline_modules.tool.saving.SavingModule"),
            ("video_rendering", "video_rendering", "temporal.pipeline_modules.tool.video_rendering.VideoRenderingModule"),
        ]:
            if (module := tree.find(f"*[@key='pipeline']/*[@key='modules']/*[@key='{src_key}']")) is not None:
                module.set("key", dst_key)
                module.set("type", dst_type)

            if (module_id := tree.find("*[@key='iteration']/*[@key='module_id']")) is not None and module_id.text == src_key:
                module_id.text = dst_key

        if (version := tree.find("*[@key='version']")) is not None:
            version.text = str(self.version)

        ET.indent(tree)
        tree.write(data_path, "utf-8")

        return True
