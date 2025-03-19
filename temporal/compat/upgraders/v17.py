import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Optional

from temporal.compat.upgrader import Upgrader
from temporal.utils.fs import load_json, load_text, save_text


class _(Upgrader):
    version = 17

    def upgrade(self, path: Path) -> bool:
        def elem(parent: ET.Element, key: str, type: str, source_dict: Optional[dict[str, Any]] = None, fallback: Any = "") -> ET.Element:
            attrs = {
                "key": key or None,
                "type": type or None,
            }

            result = ET.SubElement(parent, "object", {k: v for k, v in attrs.items() if v is not None})

            if source_dict:
                result.text = str(source_dict.get(key, fallback))

            return result

        version_path = path / "project" / "version.txt"

        if int(load_text(version_path, "0")) != self.previous_version:
            return False

        # NOTE: Session
        j_data = load_json(path / "project" / "session" / "parameters.json", {})

        j_options = j_data["shared_params"]
        j_processing = j_data["generation_params"]
        j_controlnet_units = j_data["controlnet_params"]
        j_ext_data = j_data["extension_params"]

        tree = ET.ElementTree(ET.Element("object", {"type": "temporal.session.Session"}))

        root = tree.getroot()

        options = elem(root, "options", "modules.options.Options")
        elem(options, "sd_model_checkpoint", "str", j_options)
        elem(options, "sd_vae", "str", j_options)
        elem(options, "CLIP_stop_at_last_layers", "int", j_options)
        elem(options, "always_discard_next_to_last_sigma", "bool", j_options)

        processing = elem(root, "processing", "modules.processing.StableDiffusionProcessingImg2Img")
        elem(processing, "prompt", "str", j_processing)
        elem(processing, "negative_prompt", "str", j_processing)

        init_images = elem(processing, "init_images", "list")

        for j_data in j_processing["init_images"]["data"]:
            elem(init_images, "", "PIL.Image.Image").text = str(j_data.get("filename", ""))

        elem(processing, "image_mask", "PIL.Image.Image" if j_processing["image_mask"] else "NoneType", j_processing)
        elem(processing, "resize_mode", "int", j_processing)
        elem(processing, "mask_blur_x", "int", j_processing)
        elem(processing, "mask_blur_y", "int", j_processing)
        elem(processing, "inpainting_mask_invert", "int", j_processing)
        elem(processing, "inpainting_fill", "int", j_processing)
        elem(processing, "inpaint_full_res", "int", j_processing)
        elem(processing, "inpaint_full_res_padding", "int", j_processing)
        elem(processing, "sampler_name", "str", j_processing)
        elem(processing, "steps", "int", j_processing)
        elem(processing, "refiner_checkpoint", "str" if j_processing["refiner_checkpoint"] else "NoneType", j_processing)
        elem(processing, "refiner_switch_at", "float" if j_processing["refiner_switch_at"] else "NoneType", j_processing)
        elem(processing, "width", "int", j_processing)
        elem(processing, "height", "int", j_processing)
        elem(processing, "cfg_scale", "float", j_processing)
        elem(processing, "denoising_strength", "float", j_processing)
        elem(processing, "seed", "int", j_processing)
        elem(processing, "seed_enable_extras", "bool", j_processing)
        elem(processing, "subseed", "int", j_processing)
        elem(processing, "subseed_strength", "float", j_processing)
        elem(processing, "seed_resize_from_w", "int", j_processing)
        elem(processing, "seed_resize_from_h", "int", j_processing)

        controlnet_units = elem(root, "controlnet_units", "list" if j_controlnet_units else "NoneType")

        for j_unit in j_controlnet_units:
            unit = elem(controlnet_units, "", "temporal.interop.ControlNetUnitWrapper")

            elem(unit, "instance.enabled", "bool").text = str(j_unit.get("enabled", False))
            elem(unit, "instance.module", "str").text = str(j_unit.get("module", "none"))
            elem(unit, "instance.model", "str").text = str(j_unit.get("model", "None"))
            elem(unit, "instance.weight", "float").text = str(j_unit.get("weight", 1.0))

            image = elem(unit, "instance.image", "dict" if j_unit["image"]["data"] else "NoneType")

            for j_image in j_unit["image"]["data"].items():
                elem(image, "image", "numpy.ndarray").text = str(j_image.get("filename", ""))
                elem(image, "mask", "numpy.ndarray").text = str(j_image.get("filename", ""))

            elem(unit, "instance.resize_mode", "str").text = str(j_unit.get("resize_mode", "Crop and Resize"))
            elem(unit, "instance.low_vram", "bool").text = str(j_unit.get("low_vram", False))
            elem(unit, "instance.processor_res", "int").text = str(j_unit.get("processor_res", -1))
            elem(unit, "instance.threshold_a", "float").text = str(j_unit.get("threshold_a", -1.0))
            elem(unit, "instance.threshold_b", "float").text = str(j_unit.get("threshold_b", -1.0))
            elem(unit, "instance.guidance_start", "float").text = str(j_unit.get("guidance_start", 0.0))
            elem(unit, "instance.guidance_end", "float").text = str(j_unit.get("guidance_end", 1.0))
            elem(unit, "instance.pixel_perfect", "bool").text = str(j_unit.get("pixel_perfect", False))
            elem(unit, "instance.control_mode", "str").text = str(j_unit.get("control_mode", "Balanced"))

        ext_data = elem(root, "ext_data", "temporal.data.ExtensionData")

        output = elem(ext_data, "output", "temporal.data.OutputParams")
        elem(output, "save_every_nth_frame", "int").text = str(j_ext_data.get("save_every_nth_frame", 1))
        elem(output, "archive_mode", "bool").text = str(j_ext_data.get("archive_mode", False))

        initial_noise = elem(ext_data, "initial_noise", "temporal.data.InitialNoiseParams")
        elem(initial_noise, "factor", "float").text = str(j_ext_data.get("initial_noise_factor", 0.0))
        elem(initial_noise, "scale", "int").text = str(j_ext_data.get("initial_noise_scale", 1))
        elem(initial_noise, "octaves", "int").text = str(j_ext_data.get("initial_noise_octaves", 1))
        elem(initial_noise, "lacunarity", "float").text = str(j_ext_data.get("initial_noise_lacunarity", 2.0))
        elem(initial_noise, "persistence", "float").text = str(j_ext_data.get("initial_noise_persistence", 0.5))

        processing = elem(ext_data, "processing", "temporal.data.ProcessingParams")
        elem(processing, "use_sd", "bool").text = str(j_ext_data.get("use_sd", True))

        multisampling = elem(ext_data, "multisampling", "temporal.data.MultisamplingParams")
        elem(multisampling, "samples", "int").text = str(j_ext_data.get("multisampling_samples", 1))
        elem(multisampling, "batch_size", "int").text = str(j_ext_data.get("multisampling_batch_size", 1))
        elem(multisampling, "trimming", "float").text = str(j_ext_data.get("multisampling_trimming", 0.0))
        elem(multisampling, "easing", "float").text = str(j_ext_data.get("multisampling_easing", 0.0))
        elem(multisampling, "preference", "float").text = str(j_ext_data.get("multisampling_preference", 0.0))

        detailing = elem(ext_data, "detailing", "temporal.data.DetailingParams")
        elem(detailing, "enabled", "bool").text = str(j_ext_data.get("detailing_enabled", False))
        elem(detailing, "scale", "float").text = str(j_ext_data.get("detailing_scale", 1.0))
        elem(detailing, "scale_buffer", "bool").text = str(j_ext_data.get("detailing_scale_buffer", False))
        elem(detailing, "sampler", "str").text = str(j_ext_data.get("detailing_sampler", "Euler a"))
        elem(detailing, "steps", "int").text = str(j_ext_data.get("detailing_steps", 15))
        elem(detailing, "denoising_strength", "float").text = str(j_ext_data.get("detailing_denoising_strength", 0.2))

        frame_merging = elem(ext_data, "frame_merging", "temporal.data.FrameMergingParams")
        elem(frame_merging, "frames", "int").text = str(j_ext_data.get("frame_merging_frames", 1))
        elem(frame_merging, "trimming", "float").text = str(j_ext_data.get("frame_merging_trimming", 0.0))
        elem(frame_merging, "easing", "float").text = str(j_ext_data.get("frame_merging_easing", 0.0))
        elem(frame_merging, "preference", "float").text = str(j_ext_data.get("frame_merging_preference", 0.0))

        filtering = elem(ext_data, "filtering", "temporal.data.ImageFilteringParams")

        filter_order = elem(filtering, "filter_order", "list")

        for j_module in j_ext_data["image_filtering_order"]["data"]:
            elem(filter_order, "", "str").text = j_module

        filter_data = elem(filtering, "filter_data", "dict")

        for filter_name in (
            "blurring",
            "color_balancing",
            "color_correction",
            "color_overlay",
            "custom_code",
            "image_overlay",
            "median",
            "morphology",
            "noise_compression",
            "noise_overlay",
            "palettization",
            "sharpening",
            "symmetry",
            "transformation",
        ):
            filter = elem(filter_data, filter_name, "temporal.data.ImageFilterParams")
            elem(filter, "enabled", "bool").text = str(j_ext_data.get(f"{filter_name}_enabled", False))
            elem(filter, "amount", "float").text = str(j_ext_data.get(f"{filter_name}_amount", 1.0))
            elem(filter, "amount_relative", "bool").text = str(j_ext_data.get(f"{filter_name}_amount_relative", False))
            elem(filter, "blend_mode", "str").text = str(j_ext_data.get(f"{filter_name}_blend_mode", "normal"))

            params = elem(filter, "params", "types.SimpleNamespace")

            mask = elem(filter, "mask", "temporal.data.MaskParams")

            if j_ext_data[f"{filter_name}_mask"]:
                elem(mask, "image", "PIL.Image.Image").text = str(j_ext_data.get(f"{filter_name}_mask", {}).get("filename", ""))
            else:
                elem(mask, "image", "NoneType")

            elem(mask, "normalized", "bool").text = str(j_ext_data.get(f"{filter_name}_mask_normalized", False))
            elem(mask, "inverted", "bool").text = str(j_ext_data.get(f"{filter_name}_mask_inverted", False))
            elem(mask, "blurring", "float").text = str(j_ext_data.get(f"{filter_name}_mask_blurring", 0.0))

            if filter_name == "blurring":
                elem(params, "radius", "float").text = str(j_ext_data.get(f"{filter_name}_radius", 0.0))

            elif filter_name == "color_balancing":
                elem(params, "brightness", "float").text = str(j_ext_data.get(f"{filter_name}_brightness", 1.0))
                elem(params, "contrast", "float").text = str(j_ext_data.get(f"{filter_name}_contrast", 1.0))
                elem(params, "saturation", "float").text = str(j_ext_data.get(f"{filter_name}_saturation", 1.0))

            elif filter_name == "color_correction":
                if j_ext_data[f"{filter_name}_image"]:
                    elem(params, "image", "PIL.Image.Image").text = str(j_ext_data.get(f"{filter_name}_image", {}).get("filename", ""))
                else:
                    elem(params, "image", "NoneType")

                elem(params, "normalize_contrast", "bool").text = str(j_ext_data.get(f"{filter_name}_normalize_contrast", False))
                elem(params, "equalize_histogram", "bool").text = str(j_ext_data.get(f"{filter_name}_equalize_histogram", False))

            elif filter_name == "color_overlay":
                elem(params, "color", "str").text = str(j_ext_data.get(f"{filter_name}_color", "#ffffff"))

            elif filter_name == "custom_code":
                elem(params, "code", "str").text = str(j_ext_data.get(f"{filter_name}_code", ""))

            elif filter_name == "image_overlay":
                if j_ext_data[f"{filter_name}_image"]:
                    elem(params, "image", "PIL.Image.Image").text = str(j_ext_data.get(f"{filter_name}_image", {}).get("filename", ""))
                else:
                    elem(params, "image", "NoneType")

                elem(params, "blurring", "float").text = str(j_ext_data.get(f"{filter_name}_blurring", 0.0))

            elif filter_name == "median":
                elem(params, "radius", "int").text = str(j_ext_data.get(f"{filter_name}_radius", 0))
                elem(params, "percentile", "float").text = str(j_ext_data.get(f"{filter_name}_percentile", 50.0))

            elif filter_name == "morphology":
                elem(params, "mode", "str").text = str(j_ext_data.get(f"{filter_name}_mode", "erosion"))
                elem(params, "radius", "int").text = str(j_ext_data.get(f"{filter_name}_radius", 0))

            elif filter_name == "noise_compression":
                elem(params, "constant", "float").text = str(j_ext_data.get(f"{filter_name}_constant", 0.0))
                elem(params, "adaptive", "float").text = str(j_ext_data.get(f"{filter_name}_adaptive", 0.0))

            elif filter_name == "noise_overlay":
                elem(params, "scale", "int").text = str(j_ext_data.get(f"{filter_name}_scale", 1))
                elem(params, "octaves", "int").text = str(j_ext_data.get(f"{filter_name}_octaves", 1))
                elem(params, "lacunarity", "float").text = str(j_ext_data.get(f"{filter_name}_lacunarity", 2.0))
                elem(params, "persistence", "float").text = str(j_ext_data.get(f"{filter_name}_persistence", 0.5))
                elem(params, "seed", "int").text = str(j_ext_data.get(f"{filter_name}_seed", 0))
                elem(params, "use_dynamic_seed", "bool").text = str(j_ext_data.get(f"{filter_name}_use_dynamic_seed", False))

            elif filter_name == "palettization":
                if j_ext_data[f"{filter_name}_palette"]:
                    elem(params, "palette", "PIL.Image.Image").text = str(j_ext_data.get(f"{filter_name}_palette", {}).get("filename", ""))
                else:
                    elem(params, "palette", "NoneType")

                elem(params, "stretch", "bool").text = str(j_ext_data.get(f"{filter_name}_stretch", False))
                elem(params, "dithering", "bool").text = str(j_ext_data.get(f"{filter_name}_dithering", False))

            elif filter_name == "sharpening":
                elem(params, "strength", "float").text = str(j_ext_data.get(f"{filter_name}_strength", 0.0))
                elem(params, "radius", "float").text = str(j_ext_data.get(f"{filter_name}_radius", 0.0))

            elif filter_name == "symmetry":
                elem(params, "horizontal", "bool").text = str(j_ext_data.get(f"{filter_name}_horizontal", False))
                elem(params, "vertical", "bool").text = str(j_ext_data.get(f"{filter_name}_vertical", False))

            elif filter_name == "transformation":
                elem(params, "translation_x", "float").text = str(j_ext_data.get(f"{filter_name}_translation_x", 0.0))
                elem(params, "translation_y", "float").text = str(j_ext_data.get(f"{filter_name}_translation_y", 0.0))
                elem(params, "rotation", "float").text = str(j_ext_data.get(f"{filter_name}_rotation", 0.0))
                elem(params, "scaling", "float").text = str(j_ext_data.get(f"{filter_name}_scaling", 1.0))

        ET.indent(tree)
        tree.write(path / "project" / "session" / "data.xml", "utf-8")

        # NOTE: Buffer
        j_data = load_json(path / "project" / "buffer" / "data.json", {})

        tree = ET.ElementTree(ET.Element("object", {"type": "temporal.image_buffer.ImageBuffer"}))

        root = tree.getroot()
        elem(root, "array", "numpy.ndarray").text = str(j_data.get("array", {}).get("filename", ""))
        elem(root, "last_index", "int").text = str(j_data.get("last_index", 0))

        ET.indent(tree)
        tree.write(path / "project" / "buffer" / "data.xml", "utf-8")

        save_text(version_path, str(self.version))

        return True
