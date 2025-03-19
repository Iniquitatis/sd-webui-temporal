import xml.etree.ElementTree as ET
from pathlib import Path
from shutil import copy2, rmtree
from typing import Any, Optional

from temporal.compat.upgrader import Upgrader
from temporal.utils.fs import load_text, save_text


class _(Upgrader):
    version = 18

    def upgrade(self, path: Path) -> bool:
        def obj(parent: Optional[ET.Element], key: str = "", type: str = "", text: str = "") -> Optional[ET.Element]:
            if parent is None:
                return None

            attrs = {
                "key": key or None,
                "type": type or None,
            }

            result = ET.SubElement(parent, "object", {k: v for k, v in attrs.items() if v is not None})

            if text:
                result.text = text

            return result

        def find(parent: Optional[ET.Element], key: str | list[str]) -> Optional[ET.Element]:
            if parent is None:
                return None

            if isinstance(key, str):
                key = [key]

            return parent.find("/".join(f"object[@key='{x}']" for x in key))

        def move(elem: Optional[ET.Element], old_parent: Optional[ET.Element], new_parent: Optional[ET.Element]) -> None:
            if elem is None or old_parent is None or new_parent is None:
                return None

            new_parent.append(elem)
            old_parent.remove(elem)

        def override(elem: Optional[ET.Element], **kwargs: Any) -> None:
            if elem is None:
                return

            for key, value in kwargs.items():
                elem.set(key, value)

        def value(parent: Optional[ET.Element], key: str | list[str], fallback: str = "") -> str:
            if parent is None:
                return ""

            if (elem := find(parent, key)) is not None:
                return elem.text or fallback
            else:
                return fallback

        version_path = path / "project" / "version.txt"
        session_data_path = path / "project" / "session" / "data.xml"
        buffer_path = path / "project" / "buffer"

        if int(load_text(version_path, "0")) != self.previous_version:
            return False

        tree = ET.ElementTree(file = session_data_path)

        root = tree.getroot()

        ext_data = find(root, "ext_data")

        initial_noise = find(ext_data, "initial_noise")
        override(initial_noise, type = "temporal.session.InitialNoiseParams")
        move(initial_noise, ext_data, root)

        pipeline = obj(root, "pipeline", "temporal.pipeline.Pipeline")

        module_order = obj(pipeline, "module_order", "list")
        obj(module_order, "", "str", "image_filtering")
        obj(module_order, "", "str", "processing")
        obj(module_order, "", "str", "detailing")
        obj(module_order, "", "str", "frame_merging")
        obj(module_order, "", "str", "saving")
        obj(module_order, "", "str", "measuring")
        obj(module_order, "", "str", "dampening")
        obj(module_order, "", "str", "video_rendering")

        modules = obj(pipeline, "modules", "dict")

        dampening = obj(modules, "dampening", "temporal.pipeline_modules.DampeningModule")
        obj(dampening, "enabled", "bool", "False")
        obj(dampening, "preview", "bool", "True")
        obj(dampening, "rate", "int", "1")
        obj(dampening, "buffer", "NoneType")

        detailing = find(ext_data, "detailing")
        override(detailing, type = "temporal.pipeline_modules.DetailingModule")
        move(detailing, ext_data, modules)
        obj(detailing, "preview", "bool", "True")

        frame_merging = find(ext_data, "frame_merging")
        override(frame_merging, type = "temporal.pipeline_modules.FrameMergingModule")
        move(frame_merging, ext_data, modules)
        obj(frame_merging, "enabled", "bool", "True" if value(frame_merging, "frames", "1") != "1" else "False")
        obj(frame_merging, "preview", "bool", "True")
        obj(frame_merging, "buffer_scale", "float", value(detailing, "scale", "1.0") if value(detailing, "scale_buffer", "") == "True" else "1.0")

        if (buffer_data_path := buffer_path / "data.xml").exists():
            buffer_tree = ET.ElementTree(file = buffer_data_path)
            buffer = buffer_tree.getroot()
            override(buffer, key = "buffer")

            if frame_merging is not None:
                frame_merging.append(buffer)

            copy2(buffer_path / value(buffer, "array"), path / "project" / "session")
        else:
            obj(frame_merging, "buffer", "temporal.image_buffer.ImageBuffer")

        image_filtering = obj(modules, "image_filtering", "temporal.pipeline_modules.ImageFilteringModule")
        obj(image_filtering, "enabled", "bool", "True")
        obj(image_filtering, "preview", "bool", "True")

        measuring = obj(modules, "measuring", "temporal.pipeline_modules.MeasuringModule")
        obj(measuring, "enabled", "bool", "False")
        obj(measuring, "preview", "bool", "True")
        obj(measuring, "plot_every_nth_frame", "int", "10")
        obj(measuring, "metrics", "temporal.metrics.Metrics")

        processing = find(ext_data, "multisampling")
        override(processing, key = "processing", type = "temporal.pipeline_modules.ProcessingModule")
        move(processing, ext_data, modules)
        obj(processing, "enabled", "bool", value(ext_data, ["processing", "use_sd"], "False"))
        obj(processing, "preview", "bool", "True")

        saving = find(ext_data, "output")
        override(saving, key = "saving", type = "temporal.pipeline_modules.SavingModule")
        move(saving, ext_data, modules)
        obj(saving, "enabled", "bool", "True")
        obj(saving, "preview", "bool", "True")
        obj(saving, "scale", "float", "1.0")
        obj(saving, "save_final", "bool", "False")

        video_rendering = obj(modules, "video_rendering", "temporal.pipeline_modules.VideoRenderingModule")
        obj(video_rendering, "enabled", "bool", "False")
        obj(video_rendering, "preview", "bool", "True")
        obj(video_rendering, "render_draft_every_nth_frame", "int", "100")
        obj(video_rendering, "render_final_every_nth_frame", "int", "1000")
        obj(video_rendering, "render_draft_on_finish", "bool", "False")
        obj(video_rendering, "render_final_on_finish", "bool", "False")

        image_filterer = find(ext_data, "filtering")
        override(image_filterer, key = "image_filterer", type = "temporal.image_filterer.ImageFilterer")
        move(image_filterer, ext_data, root)

        filters = find(image_filterer, "filter_data")
        override(filters, key = "filters")

        for key, new_type in (
            ("blurring", "temporal.image_filters.BlurringFilter"),
            ("color_balancing", "temporal.image_filters.ColorBalancingFilter"),
            ("color_correction", "temporal.image_filters.ColorCorrectionFilter"),
            ("color_overlay", "temporal.image_filters.ColorOverlayFilter"),
            ("custom_code", "temporal.image_filters.CustomCodeFilter"),
            ("image_overlay", "temporal.image_filters.ImageOverlayFilter"),
            ("median", "temporal.image_filters.MedianFilter"),
            ("morphology", "temporal.image_filters.MorphologyFilter"),
            ("noise_compression", "temporal.image_filters.NoiseCompressionFilter"),
            ("noise_overlay", "temporal.image_filters.NoiseOverlayFilter"),
            ("palettization", "temporal.image_filters.PalettizationFilter"),
            ("sharpening", "temporal.image_filters.SharpeningFilter"),
            ("symmetry", "temporal.image_filters.SymmetryFilter"),
            ("transformation", "temporal.image_filters.TransformationFilter"),
        ):
            filter = find(filters, key)
            override(filter, type = new_type)

            if filter is None:
                continue

            mask = find(filter, "mask")
            override(mask, type = "temporal.image_mask.ImageMask")

            for params in filter.findall("*[@key='params']"):
                for param in list(params):
                    move(param, params, filter)

                filter.remove(params)

        if ext_data is not None:
            root.remove(ext_data)

        ET.indent(tree)
        tree.write(session_data_path, "utf-8")

        rmtree(buffer_path)

        save_text(version_path, str(self.version))

        return True
