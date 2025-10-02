import xml.etree.ElementTree as ET
from pathlib import Path

from modules.compat.upgrader import Upgrader


class _(Upgrader):
    version = 42

    def upgrade(self, path: Path) -> bool:
        data_path = path / "project" / "data.xml"

        if not data_path.exists():
            return False

        tree = ET.ElementTree(file = data_path)

        if tree.findtext("*[@key='version']", "0") != str(self.previous_version):
            return False

        for modules in tree.iterfind("*[@key='pipeline']/*[@key='modules']"):
            modules.set("type", "list")

            for module in modules:
                module.attrib.pop("key")

        classes = {
            "blurring": "filtering.blurring.BlurringFilter",
            "color_balancing": "filtering.color_balancing.ColorBalancingFilter",
            "color_correction": "filtering.color_correction.ColorCorrectionFilter",
            "custom_code": "filtering.custom_code.CustomCodeFilter",
            "median": "filtering.median.MedianFilter",
            "morphology": "filtering.morphology.MorphologyFilter",
            "noise_compression": "filtering.noise_compression.NoiseCompressionFilter",
            "palettization": "filtering.palettization.PalettizationFilter",
            "pixelization": "filtering.pixelization.PixelizationFilter",
            "sharpening": "filtering.sharpening.SharpeningFilter",
            "symmetry": "filtering.symmetry.SymmetryFilter",
            "transformation": "filtering.transformation.TransformationFilter",

            "color_level_mean_measuring": "measuring.color_level_mean.ColorLevelMeanMeasuringModule",
            "color_level_sigma_measuring": "measuring.color_level_sigma.ColorLevelSigmaMeasuringModule",
            "luminance_mean_measuring": "measuring.luminance_mean.LuminanceMeanMeasuringModule",
            "luminance_sigma_measuring": "measuring.luminance_sigma.LuminanceSigmaMeasuringModule",
            "noise_sigma_measuring": "measuring.noise_sigma.NoiseSigmaMeasuringModule",

            "averaging": "temporal.averaging.AveragingModule",
            "interpolation": "temporal.interpolation.InterpolationModule",
            "limiting": "temporal.limiting.LimitingModule",
            "random_sampling": "temporal.random_sampling.RandomSamplingModule",

            "color_painting": "painting.color.ColorPaintingModule",
            "gradient_painting": "painting.gradient.GradientPaintingModule",
            "image_painting": "painting.image.ImagePaintingModule",
            "noise_painting": "painting.noise.NoisePaintingModule",
            "pattern_painting": "painting.pattern.PatternPaintingModule",

            "saving": "tool.saving.SavingModule",
            "video_rendering": "tool.video_rendering.VideoRenderingModule",

            "detailing": "neural.detailing.DetailingModule",
            "processing": "neural.processing.ProcessingModule",
        }

        if ((module_id := tree.find("*[@key='iteration']/*[@key='module_id']")) is not None and
            module_id.text is not None and
            module_id.text != "None"):
            module_id.text = f"temporal.pipeline_modules.{classes[module_id.text]}"

        if (version := tree.find("*[@key='version']")) is not None:
            version.text = str(self.version)

        ET.indent(tree)
        tree.write(data_path, "utf-8")

        return True
