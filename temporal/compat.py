import xml.etree.ElementTree as ET
from abc import abstractmethod
from pathlib import Path
from shutil import copy2, rmtree
from typing import Any, Optional, Type

import numpy as np

from temporal.meta.registerable import Registerable
from temporal.utils import logging
from temporal.utils.fs import ensure_directory_exists, load_json, load_text, move_entry, remove_entry, save_json, save_text
from temporal.utils.image import load_image, pil_to_np
from temporal.utils.numpy import load_array, save_array


UPGRADERS: list[Type["Upgrader"]] = []


def get_latest_version() -> int:
    return max(x.version for x in UPGRADERS)


def upgrade_project(path: Path) -> None:
    last_version = 0

    for cls in UPGRADERS:
        upgrader = cls()

        if upgrader.upgrade(path):
            last_version = upgrader.version

    if last_version:
        logging.info(f"Upgraded project to version {last_version}")


class Upgrader(Registerable, abstract = True):
    store = UPGRADERS

    version: int = -1

    @abstractmethod
    def upgrade(self, path: Path) -> bool:
        raise NotImplementedError

    @property
    def previous_version(self) -> int:
        return max(x.version for x in UPGRADERS if x.version < self.version)


class _(Upgrader):
    version = 1

    def upgrade(self, path: Path) -> bool:
        def upgrade_value(value: Any) -> Any:
            if isinstance(value, list):
                return {"type": "list", "data": [upgrade_value(x) for x in value]}
            elif isinstance(value, dict):
                if "im_type" in value:
                    return {"type": value["im_type"], "filename": value["filename"]}
                else:
                    return {"type": "dict", "data": {k: upgrade_value(v) for k, v in value.items()}}
            else:
                return value

        def upgrade_values(d: dict[str, Any]) -> dict[str, Any]:
            return {k: upgrade_value(v) for k, v in d.items()}

        version_path = path / "session" / "version.txt"
        params_path = path / "session" / "parameters.json"

        if "im_type" not in load_text(params_path, ""):
            return False

        data = load_json(params_path, {})

        data["shared_params"] = upgrade_values(data.get("shared_params", {}))
        data["generation_params"] = upgrade_values(data.get("generation_params", {}))

        for i, unit_data in enumerate(data.get("controlnet_params", [])):
            data["controlnet_params"][i] = upgrade_values(unit_data)

        data["extension_params"] = upgrade_values(data.get("extension_params", {}))

        save_json(params_path, data)
        save_text(version_path, str(self.version))

        return True


class _(Upgrader):
    version = 2

    def upgrade(self, path: Path) -> bool:
        version_path = path / "session" / "version.txt"
        params_path = path / "session" / "parameters.json"

        if int(load_text(version_path, "0")) != self.previous_version:
            return False

        data = load_json(params_path, {})

        ext_params = data["extension_params"]

        for before, after in [
            ("normalize_contrast", "color_correction_normalize_contrast"),
            ("brightness", "color_balancing_brightness"),
            ("contrast", "color_balancing_contrast"),
            ("saturation", "color_balancing_saturation"),
            ("noise_relative", "noise_amount_relative"),
            ("modulation_relative", "modulation_amount_relative"),
            ("tinting_relative", "tinting_amount_relative"),
            ("sharpening_amount", "sharpening_strength"),
            ("sharpening_relative", "sharpening_amount_relative"),
            ("translation_x", "transformation_translation_x"),
            ("translation_y", "transformation_translation_y"),
            ("rotation", "transformation_rotation"),
            ("scaling", "transformation_scaling"),
            ("symmetrize", "symmetry_enabled"),
            ("custom_code", "custom_code_code"),
        ]:
            ext_params[after] = ext_params.pop(before)

        for key in [
            "noise_compression_amount",
            "color_correction_amount",
            "color_balancing_amount",
            "sharpening_amount",
            "transformation_amount",
            "symmetry_amount",
            "blurring_amount",
            "custom_code_amount",
        ]:
            ext_params[key] = 1.0

        save_json(params_path, data)
        save_text(version_path, str(self.version))

        return True


class _(Upgrader):
    version = 3

    def upgrade(self, path: Path) -> bool:
        version_path = path / "session" / "version.txt"

        if int(load_text(version_path, "0")) != self.previous_version:
            return False

        if frames := sorted(path.glob("*.png"), key = lambda x: int(x.stem)):
            copy2(frames[-1], ensure_directory_exists(path / "session" / "buffer") / "001.png")

        save_text(version_path, str(self.version))

        return True


class _(Upgrader):
    version = 4

    def upgrade(self, path: Path) -> bool:
        def upgrade_value(value: Any) -> Any:
            if isinstance(value, dict):
                type = value.get("type", None)

                if type == "list":
                    return {"type": "list", "data": [upgrade_value(x) for x in value["data"]]}
                elif type == "dict":
                    return {"type": "dict", "data": {k: upgrade_value(v) for k, v in value["data"].items()}}
                elif type == "np":
                    im_path = path / "session" / value["filename"]
                    arr_path = im_path.with_suffix(".npy")
                    np.save(arr_path, np.array(load_image(im_path)))
                    im_path.unlink()
                    return {"type": "np", "filename": arr_path.name}
                else:
                    return value
            else:
                return value

        def upgrade_values(d: dict[str, Any]) -> dict[str, Any]:
            return {k: upgrade_value(v) for k, v in d.items()}

        version_path = path / "session" / "version.txt"
        params_path = path / "session" / "parameters.json"

        if int(load_text(version_path, "0")) != self.previous_version:
            return False

        data = load_json(params_path, {})

        data["shared_params"] = upgrade_values(data.get("shared_params", {}))
        data["generation_params"] = upgrade_values(data.get("generation_params", {}))

        for i, unit_data in enumerate(data.get("controlnet_params", [])):
            data["controlnet_params"][i] = upgrade_values(unit_data)

        data["extension_params"] = upgrade_values(data.get("extension_params", {}))

        save_json(params_path, data)
        save_text(version_path, str(self.version))

        return True


class _(Upgrader):
    version = 5

    def upgrade(self, path: Path) -> bool:
        def upgrade_value(value: Any) -> Any:
            if isinstance(value, dict):
                type = value.get("type", None)

                if type == "list":
                    return {"type": "list", "data": [upgrade_value(x) for x in value["data"]]}
                elif type == "dict":
                    return {"type": "dict", "data": {k: upgrade_value(v) for k, v in value["data"].items()}}
                elif type == "np":
                    arr_path = path / "session" / value["filename"]
                    arrz_path = arr_path.with_suffix(".npz")
                    np.savez_compressed(arrz_path, np.load(arr_path))
                    arr_path.unlink()
                    return {"type": "np", "filename": arrz_path.name}
                else:
                    return value
            else:
                return value

        def upgrade_values(d: dict[str, Any]) -> dict[str, Any]:
            return {k: upgrade_value(v) for k, v in d.items()}

        version_path = path / "session" / "version.txt"
        params_path = path / "session" / "parameters.json"
        buffer_dir = path / "session" / "buffer"

        if int(load_text(version_path, "0")) != self.previous_version:
            return False

        data = load_json(params_path, {})

        data["shared_params"] = upgrade_values(data.get("shared_params", {}))
        data["generation_params"] = upgrade_values(data.get("generation_params", {}))

        for i, unit_data in enumerate(data.get("controlnet_params", [])):
            data["controlnet_params"][i] = upgrade_values(unit_data)

        data["extension_params"] = upgrade_values(data.get("extension_params", {}))

        save_json(params_path, data)

        image_paths = sorted(buffer_dir.glob("*.png"), key = lambda x: int(x.stem))

        np.savez_compressed(buffer_dir / "buffer.npz", np.stack([
            pil_to_np(load_image(x))
            for x in image_paths
        ], axis = 0))

        for path in image_paths:
            path.unlink()

        save_json(buffer_dir / "data.json", {
            "array": {
                "type": "np",
                "filename": "buffer.npz",
            },
            "last_index": 0,
        })

        save_text(version_path, str(self.version))

        return True


class _(Upgrader):
    version = 6

    def upgrade(self, path: Path) -> bool:
        version_path = path / "session" / "version.txt"
        params_path = path / "session" / "parameters.json"

        if int(load_text(version_path, "0")) != self.previous_version:
            return False

        data = load_json(params_path, {})

        ext_params = data["extension_params"]
        ext_params.update({
            "multisampling_samples": ext_params.pop("image_samples", 1),
            "multisampling_batch_size": ext_params.pop("batch_size", 1),
            "multisampling_algorithm": "mean",
            "multisampling_easing": 0.0,
            "frame_merging_frames": ext_params.pop("merged_frames", 1),
            "frame_merging_algorithm": "mean",
            "frame_merging_easing": ext_params.pop("merged_frames_easing", 0.0),
        })

        save_json(params_path, data)
        save_text(version_path, str(self.version))

        return True


class _(Upgrader):
    version = 7

    def upgrade(self, path: Path) -> bool:
        version_path = path / "session" / "version.txt"
        params_path = path / "session" / "parameters.json"

        if int(load_text(version_path, "0")) != self.previous_version:
            return False

        data = load_json(params_path, {})

        ext_params = data["extension_params"]
        ext_params["preprocessing_order"] = {
            "type": "list",
            "data": [
                "noise_compression",
                "color_correction",
                "color_balancing",
                "noise",
                "modulation",
                "tinting",
                "sharpening",
                "transformation",
                "symmetry",
                "blurring",
                "custom_code",
            ],
        }

        save_json(params_path, data)
        save_text(version_path, str(self.version))

        return True


class _(Upgrader):
    version = 8

    def upgrade(self, path: Path) -> bool:
        version_path = path / "session" / "version.txt"
        params_path = path / "session" / "parameters.json"

        if int(load_text(version_path, "0")) != self.previous_version:
            return False

        data = load_json(params_path, {})

        ext_params = data["extension_params"]

        for key in ["multisampling_algorithm", "frame_merging_algorithm"]:
            if ext_params[key] == "mean":
                ext_params[key] = "arithmetic_mean"

        save_json(params_path, data)
        save_text(version_path, str(self.version))

        return True


class _(Upgrader):
    version = 9

    def upgrade(self, path: Path) -> bool:
        version_path = path / "session" / "version.txt"
        params_path = path / "session" / "parameters.json"

        if int(load_text(version_path, "0")) != self.previous_version:
            return False

        data = load_json(params_path, {})

        ext_params = data["extension_params"]

        for feature in ["multisampling", "frame_merging"]:
            if (algo := ext_params.pop(f"{feature}_algorithm")) != "median":
                ext_params[f"{feature}_preference"] = {
                    "harmonic_mean": -2.0,
                    "geometric_mean": -1.0,
                    "arithmetic_mean": 0.0,
                    "root_mean_square": 1.0,
                }[algo]
            else:
                ext_params[f"{feature}_trimming"] = 0.5
                ext_params[f"{feature}_preference"] = 1.0

        save_json(params_path, data)
        save_text(version_path, str(self.version))

        return True


class _(Upgrader):
    version = 10

    def upgrade(self, path: Path) -> bool:
        version_path = path / "session" / "version.txt"
        params_path = path / "session" / "parameters.json"

        if int(load_text(version_path, "0")) != self.previous_version:
            return False

        data = load_json(params_path, {})

        ext_params = data["extension_params"]
        ext_params.update({
            "initial_noise_factor": float(ext_params.pop("noise_for_first_frame")),
            "initial_noise_scale": 1,
            "initial_noise_octaves": 1,
            "initial_noise_lacunarity": 2.0,
            "initial_noise_persistence": 0.5,
        })

        save_json(params_path, data)
        save_text(version_path, str(self.version))

        return True


class _(Upgrader):
    version = 11

    def upgrade(self, path: Path) -> bool:
        version_path = path / "session" / "version.txt"
        params_path = path / "session" / "parameters.json"

        if int(load_text(version_path, "0")) != self.previous_version:
            return False

        data = load_json(params_path, {})

        ext_params = data["extension_params"]
        ext_params.update({
            "blurring_blend_mode": "normal",
            "color_balancing_blend_mode": "normal",
            "color_correction_blend_mode": "normal",
            "color_overlay_amount": ext_params.pop("tinting_amount"),
            "color_overlay_amount_relative": ext_params.pop("tinting_amount_relative"),
            "color_overlay_blend_mode": ext_params.pop("tinting_mode"),
            "color_overlay_color": ext_params.pop("tinting_color"),
            "color_overlay_mask": ext_params.pop("tinting_mask"),
            "color_overlay_mask_normalized": ext_params.pop("tinting_mask_normalized"),
            "color_overlay_mask_inverted": ext_params.pop("tinting_mask_inverted"),
            "color_overlay_mask_blurring": ext_params.pop("tinting_mask_blurring"),
            "custom_code_blend_mode": "normal",
            "image_overlay_amount": ext_params.pop("modulation_amount"),
            "image_overlay_amount_relative": ext_params.pop("modulation_amount_relative"),
            "image_overlay_blend_mode": ext_params.pop("modulation_mode"),
            "image_overlay_image": ext_params.pop("modulation_image"),
            "image_overlay_blurring": ext_params.pop("modulation_blurring"),
            "image_overlay_mask": ext_params.pop("modulation_mask"),
            "image_overlay_mask_normalized": ext_params.pop("modulation_mask_normalized"),
            "image_overlay_mask_inverted": ext_params.pop("modulation_mask_inverted"),
            "image_overlay_mask_blurring": ext_params.pop("modulation_mask_blurring"),
            "median_blend_mode": "normal",
            "morphology_blend_mode": "normal",
            "noise_compression_blend_mode": "normal",
            "noise_overlay_amount": ext_params.pop("noise_amount"),
            "noise_overlay_amount_relative": ext_params.pop("noise_amount_relative"),
            "noise_overlay_blend_mode": ext_params.pop("noise_mode"),
            "noise_overlay_mask": ext_params.pop("noise_mask"),
            "noise_overlay_mask_normalized": ext_params.pop("noise_mask_normalized"),
            "noise_overlay_mask_inverted": ext_params.pop("noise_mask_inverted"),
            "noise_overlay_mask_blurring": ext_params.pop("noise_mask_blurring"),
            "palettization_blend_mode": "normal",
            "sharpening_blend_mode": "normal",
            "symmetry_blend_mode": "normal",
            "transformation_blend_mode": "normal",
        })

        save_json(params_path, data)
        save_text(version_path, str(self.version))

        return True


class _(Upgrader):
    version = 12

    def upgrade(self, path: Path) -> bool:
        version_path = path / "session" / "version.txt"
        params_path = path / "session" / "parameters.json"

        if int(load_text(version_path, "0")) != self.previous_version:
            return False

        data = load_json(params_path, {})

        ext_params = data["extension_params"]
        ext_params.update({
            "noise_overlay_scale": 1,
            "noise_overlay_octaves": 1,
            "noise_overlay_lacunarity": 2.0,
            "noise_overlay_persistence": 0.5,
            "noise_overlay_seed": 0,
            "noise_overlay_use_dynamic_seed": True,
        })

        save_json(params_path, data)
        save_text(version_path, str(self.version))

        return True


class _(Upgrader):
    version = 13

    def upgrade(self, path: Path) -> bool:
        version_path = path / "session" / "version.txt"
        params_path = path / "session" / "parameters.json"

        if int(load_text(version_path, "0")) != self.previous_version:
            return False

        data = load_json(params_path, {})

        ext_params = data["extension_params"]
        ext_params.update({
            "symmetry_horizontal": True,
            "symmetry_vertical": False,
        })

        save_json(params_path, data)
        save_text(version_path, str(self.version))

        return True


class _(Upgrader):
    version = 14

    def upgrade(self, path: Path) -> bool:
        version_path = path / "session" / "version.txt"
        params_path = path / "session" / "parameters.json"

        if int(load_text(version_path, "0")) != self.previous_version:
            return False

        data = load_json(params_path, {})

        ext_params = data["extension_params"]
        ext_params.update({
            "median_percentile": 50.0,
        })

        save_json(params_path, data)
        save_text(version_path, str(self.version))

        return True


class _(Upgrader):
    version = 15

    def upgrade(self, path: Path) -> bool:
        version_path = path / "session" / "version.txt"

        if int(load_text(version_path, "0")) != self.previous_version:
            return False

        project_data_dir = ensure_directory_exists(path / "project")
        move_entry(path / "metrics", project_data_dir / "metrics")
        move_entry(path / "session" / "buffer", project_data_dir / "buffer")
        move_entry(path / "session", project_data_dir / "session")
        move_entry(version_path, project_data_dir / "version.txt")

        save_text(project_data_dir / "version.txt", str(self.version))

        return True


class _(Upgrader):
    version = 16

    def upgrade(self, path: Path) -> bool:
        version_path = path / "project" / "version.txt"
        params_path = path / "project" / "session" / "parameters.json"

        if int(load_text(version_path, "0")) != self.previous_version:
            return False

        data = load_json(params_path, {})

        ext_params = data["extension_params"]
        ext_params.update({
            "image_filtering_order": ext_params.pop("preprocessing_order"),
        })

        save_json(params_path, data)
        save_text(version_path, str(self.version))

        return True


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


class _(Upgrader):
    version = 19

    def upgrade(self, path: Path) -> bool:
        version_path = path / "project" / "version.txt"
        session_data_path = path / "project" / "session" / "data.xml"

        if int(load_text(version_path, "0")) != self.previous_version:
            return False

        tree = ET.ElementTree(file = session_data_path)
        root = tree.getroot()

        if (modules := root.find(".//*[@key='pipeline']/*[@key='modules']")) is not None:
            limiting = ET.SubElement(modules, "object", {"key": "limiting", "type": "temporal.pipeline_modules.LimitingModule"})
            ET.SubElement(limiting, "object", {"key": "enabled", "type": "bool"}).text = "False"
            ET.SubElement(limiting, "object", {"key": "preview", "type": "bool"}).text = "True"
            ET.SubElement(limiting, "object", {"key": "mode", "type": "str"}).text = "clamp"
            ET.SubElement(limiting, "object", {"key": "max_difference", "type": "float"}).text = "1.0"
            ET.SubElement(limiting, "object", {"key": "buffer", "type": "NoneType"})

        if (module_order := root.find(".//*[@key='pipeline']/*[@key='module_order']")) is not None:
            ET.SubElement(module_order, "object", {"type": "str"}).text = "limiting"

        ET.indent(tree)
        tree.write(session_data_path, "utf-8")
        save_text(version_path, str(self.version))

        return True



class _(Upgrader):
    version = 20

    def upgrade(self, path: Path) -> bool:
        version_path = path / "project" / "version.txt"
        session_data_path = path / "project" / "session" / "data.xml"

        if int(load_text(version_path, "0")) != self.previous_version:
            return False

        tree = ET.ElementTree(file = session_data_path)

        if (frame_merging := tree.find(".//*[@key='frame_merging']")) is not None:
            if (buffer := frame_merging.find("*[@key='buffer']")) is not None:
                for elem in list(buffer):
                    frame_merging.append(elem)
                    buffer.remove(elem)

                frame_merging.remove(buffer)

            if (array := frame_merging.find("*[@key='array']")) is not None:
                array.set("key", "buffer")

        ET.indent(tree)
        tree.write(session_data_path, "utf-8")
        save_text(version_path, str(self.version))

        return True


class _(Upgrader):
    version = 21

    def upgrade(self, path: Path) -> bool:
        def convert(elem: Optional[ET.Element]) -> None:
            if elem is None or elem.text is None:
                return

            im_path = path / "project" / "session" / Path(elem.text)

            if not im_path.exists():
                return

            arr_path = im_path.with_suffix(".npz")

            im = load_image(im_path)
            save_array(pil_to_np(im), arr_path)

            elem.set("type", "numpy.ndarray")
            elem.text = arr_path.name

            im_path.unlink()

        version_path = path / "project" / "version.txt"
        session_data_path = path / "project" / "session" / "data.xml"

        if int(load_text(version_path, "0")) != self.previous_version:
            return False

        tree = ET.ElementTree(file = session_data_path)

        for elem in tree.findall(".//*[@key='mask']/*[@key='image']"):
            convert(elem)

        convert(tree.find(".//*[@key='color_correction']/*[@key='image']"))
        convert(tree.find(".//*[@key='image_overlay']/*[@key='image']"))
        convert(tree.find(".//*[@key='palettization']/*[@key='palette']"))

        ET.indent(tree)
        tree.write(session_data_path, "utf-8")
        save_text(version_path, str(self.version))

        return True


class _(Upgrader):
    version = 22

    def upgrade(self, path: Path) -> bool:
        version_path = path / "project" / "version.txt"
        session_data_path = path / "project" / "session" / "data.xml"

        if int(load_text(version_path, "0")) != self.previous_version:
            return False

        tree = ET.ElementTree(file = session_data_path)

        if (module_order := tree.find(".//*[@key='module_order']")) is not None and \
           (filter_order := tree.find(".//*[@key='filter_order']")) is not None:
            if (image_filtering_id := module_order.find(".//*[.='image_filtering']")) is not None:
                image_filtering_index = list(module_order).index(image_filtering_id)

                for i, filter_id in enumerate(filter_order):
                    module_order.insert(image_filtering_index + i, filter_id)

                module_order.remove(image_filtering_id)

            else:
                for filter_id in filter_order:
                    module_order.append(filter_id)

        if (modules := tree.find(".//*[@key='pipeline']/*[@key='modules']")) is not None and \
           (filters := tree.find(".//*[@key='image_filterer']/*[@key='filters']")) is not None:
            for filter in filters:
                modules.append(filter)

            if (image_filtering := modules.find(".//*[@key='image_filtering']")) is not None:
                modules.remove(image_filtering)

        if (image_filterer := tree.find(".//*[@key='image_filterer']")) is not None:
            tree.getroot().remove(image_filterer)

        ET.indent(tree)
        tree.write(session_data_path, "utf-8")
        save_text(version_path, str(self.version))

        return True


class _(Upgrader):
    version = 23

    def upgrade(self, path: Path) -> bool:
        version_path = path / "project" / "version.txt"
        session_data_path = path / "project" / "session" / "data.xml"

        if int(load_text(version_path, "0")) != self.previous_version:
            return False

        tree = ET.ElementTree(file = session_data_path)

        if (frame_merging := tree.find(".//*[@key='frame_merging']")) is not None:
            frame_merging.set("key", "averaging")
            frame_merging.set("type", "temporal.pipeline_modules.AveragingModule")

        if (dampening := tree.find(".//*[@key='dampening']")) is not None:
            dampening.set("key", "interpolation")
            dampening.set("type", "temporal.pipeline_modules.InterpolationModule")

        if (frame_merging_id := tree.find(".//*[@key='pipeline']/*[@key='module_order']/*[.='frame_merging']")) is not None:
            frame_merging_id.text = "averaging"

        if (dampening_id := tree.find(".//*[@key='pipeline']/*[@key='module_order']/*[.='dampening']")) is not None:
            dampening_id.text = "interpolation"

        ET.indent(tree)
        tree.write(session_data_path, "utf-8")
        save_text(version_path, str(self.version))

        return True


class _(Upgrader):
    version = 24

    def upgrade(self, path: Path) -> bool:
        def parse_frame_index(im_path: Path) -> int:
            if im_path.is_file():
                try:
                    return int(im_path.stem)
                except:
                    logging.warning(f"{im_path.stem} doesn't match the frame name format")

            return 0

        version_path = path / "project" / "version.txt"
        session_data_path = path / "project" / "session" / "data.xml"

        if int(load_text(version_path, "0")) != self.previous_version:
            return False

        tree = ET.ElementTree(file = session_data_path)
        root = tree.getroot()

        last_frame_index = max((parse_frame_index(x) for x in path.glob("*.png")), default = 0)

        iteration = ET.SubElement(root, "object", {"key": "iteration", "type": "temporal.session.IterationData"})

        images = ET.SubElement(iteration, "object", {"key": "images", "type": "list"})

        im_path = None

        if (last_frame_path := (path / f"{last_frame_index:05d}.png")).is_file():
            im_path = last_frame_path
        elif (init_image_name := tree.findtext("*[@key='processing']/*[@key='init_images']/*[@type='PIL.Image.Image']")) is not None:
            im_path = path / "project" / "session" / init_image_name

        if im_path is not None:
            arr_path = path / "project" / "session" / "last_frame.npz"

            im = load_image(im_path)
            save_array(pil_to_np(im), arr_path)

            image = ET.SubElement(images, "object", {"type": "numpy.ndarray"})
            image.text = arr_path.name

        index = ET.SubElement(iteration, "object", {"key": "index", "type": "int"})
        index.text = str(last_frame_index + 1)

        ET.SubElement(iteration, "object", {"key": "module_id", "type": "NoneType"})

        ET.indent(tree)
        tree.write(session_data_path, "utf-8")
        save_text(version_path, str(self.version))

        return True


class _(Upgrader):
    version = 25

    def upgrade(self, path: Path) -> bool:
        def upgrade_buffer(elem: ET.Element) -> None:
            if elem.text is not None and (arr_path := (path / "project" / "session" / elem.text)).is_file():
                save_array(np.expand_dims(load_array(arr_path), 0), arr_path)

        version_path = path / "project" / "version.txt"
        session_data_path = path / "project" / "session" / "data.xml"

        if int(load_text(version_path, "0")) != self.previous_version:
            return False

        tree = ET.ElementTree(file = session_data_path)

        if (pipeline := tree.find(".//*[@key='pipeline']")) is not None:
            ET.SubElement(pipeline, "object", {"key": "parallel", "type": "int"}).text = "1"

        if (processing := tree.find(".//*[@key='pipeline']/*[@key='modules']/*[@key='processing']")) is not None:
            ET.SubElement(processing, "object", {"key": "pixels_per_batch", "type": "int"}).text = "1048576"

        if (buffer := tree.find(".//*[@key='pipeline']/*[@key='modules']/*[@key='averaging']/*[@key='buffer']")) is not None:
            upgrade_buffer(buffer)

        if (buffer := tree.find(".//*[@key='pipeline']/*[@key='modules']/*[@key='interpolation']/*[@key='buffer']")) is not None:
            upgrade_buffer(buffer)

        if (buffer := tree.find(".//*[@key='pipeline']/*[@key='modules']/*[@key='limiting']/*[@key='buffer']")) is not None:
            upgrade_buffer(buffer)

        ET.indent(tree)
        tree.write(session_data_path, "utf-8")
        save_text(version_path, str(self.version))

        return True


class _(Upgrader):
    version = 26

    def upgrade(self, path: Path) -> bool:
        version_path = path / "project" / "version.txt"
        session_data_path = path / "project" / "session" / "data.xml"

        if int(load_text(version_path, "0")) != self.previous_version:
            return False

        tree = ET.ElementTree(file = session_data_path)

        if (interpolation := tree.find(".//*[@key='pipeline']/*[@key='modules']/*[@key='interpolation']")) is not None:
            if (rate := interpolation.find("*[@key='rate']")) is not None:
                rate.set("key", "blending")

            ET.SubElement(interpolation, "object", {"key": "movement", "type": "float"}).text = "0.0"
            ET.SubElement(interpolation, "object", {"key": "radius", "type": "int"}).text = "15"

        ET.indent(tree)
        tree.write(session_data_path, "utf-8")
        save_text(version_path, str(self.version))

        return True


class _(Upgrader):
    version = 27

    def upgrade(self, path: Path) -> bool:
        version_path = path / "project" / "version.txt"

        if int(load_text(version_path, "0")) != self.previous_version:
            return False

        session_dir = path / "project" / "session"

        for entry_path in session_dir.iterdir():
            move_entry(entry_path, path / "project" / entry_path.name)

        remove_entry(session_dir)
        remove_entry(version_path)

        new_root = ET.Element("object", {"type": "temporal.project.Project"})
        new_tree = ET.ElementTree(new_root)

        version = ET.SubElement(new_root, "object", {"key": "version", "type": "int"})
        version.text = str(self.version)

        data_path = path / "project" / "data.xml"

        old_tree = ET.ElementTree(file = data_path)

        session = old_tree.getroot()
        session.set("key", "session")
        new_root.append(session)

        ET.indent(new_tree)
        new_tree.write(data_path, "utf-8")

        return True


class _(Upgrader):
    version = 28

    def upgrade(self, path: Path) -> bool:
        data_path = path / "project" / "data.xml"

        if not data_path.exists():
            return False

        tree = ET.ElementTree(file = data_path)

        if tree.findtext("*[@key='version']", "0") != str(self.previous_version):
            return False

        if (initial_noise := tree.find("*[@key='session']/*[@key='initial_noise']")) is not None:
            noise = ET.SubElement(initial_noise, "object", {"key": "noise", "type": "temporal.noise.Noise"})

            if (scale := initial_noise.find("*[@key='scale']")) is not None:
                noise.append(scale)
                initial_noise.remove(scale)

            if (octaves := initial_noise.find("*[@key='octaves']")) is not None:
                noise.append(octaves)
                initial_noise.remove(octaves)

            if (lacunarity := initial_noise.find("*[@key='lacunarity']")) is not None:
                noise.append(lacunarity)
                initial_noise.remove(lacunarity)

            if (persistence := initial_noise.find("*[@key='persistence']")) is not None:
                noise.append(persistence)
                initial_noise.remove(persistence)

            ET.SubElement(noise, "object", {"key": "seed", "type": "int"}).text = "0"
            ET.SubElement(initial_noise, "object", {"key": "use_initial_seed", "type": "bool"}).text = "True"

        if (color := tree.find("*[@key='session']/*[@key='pipeline']/*[@key='modules']/*[@key='color_overlay']/*[@key='color']")) is not None:
            color.set("type", "temporal.color.Color")

            if color.text is not None:
                parts = [color.text[i:i + 2] for i in range(1, 7, 2)]

                color.text = None

                ET.SubElement(color, "object", {"key": "r", "type": "float"}).text = str(int(parts[0], 16) / 255.0)
                ET.SubElement(color, "object", {"key": "g", "type": "float"}).text = str(int(parts[1], 16) / 255.0)
                ET.SubElement(color, "object", {"key": "b", "type": "float"}).text = str(int(parts[2], 16) / 255.0)
                ET.SubElement(color, "object", {"key": "a", "type": "float"}).text = "1.0"

        if (noise_overlay := tree.find("*[@key='session']/*[@key='pipeline']/*[@key='modules']/*[@key='noise_overlay']")) is not None:
            noise = ET.SubElement(noise_overlay, "object", {"key": "noise", "type": "temporal.noise.Noise"})

            if (scale := noise_overlay.find("*[@key='scale']")) is not None:
                noise.append(scale)
                noise_overlay.remove(scale)

            if (octaves := noise_overlay.find("*[@key='octaves']")) is not None:
                noise.append(octaves)
                noise_overlay.remove(octaves)

            if (lacunarity := noise_overlay.find("*[@key='lacunarity']")) is not None:
                noise.append(lacunarity)
                noise_overlay.remove(lacunarity)

            if (persistence := noise_overlay.find("*[@key='persistence']")) is not None:
                noise.append(persistence)
                noise_overlay.remove(persistence)

            if (seed := noise_overlay.find("*[@key='seed']")) is not None:
                noise.append(seed)
                noise_overlay.remove(seed)

        if (version := tree.find("*[@key='version']")) is not None:
            version.text = str(self.version)

        ET.indent(tree)
        tree.write(data_path, "utf-8")

        return True


class _(Upgrader):
    version = 29

    def upgrade(self, path: Path) -> bool:
        data_path = path / "project" / "data.xml"

        if not data_path.exists():
            return False

        tree = ET.ElementTree(file = data_path)

        if tree.findtext("*[@key='version']", "0") != str(self.previous_version):
            return False

        for module_id in (
            "color_correction",
            "image_overlay",
        ):
            if (module := tree.find(f"*[@key='session']/*[@key='pipeline']/*[@key='modules']/*[@key='{module_id}']")) is not None:
                source = ET.SubElement(module, "object", {"key": "source", "type": "temporal.image_source.ImageSource"})
                ET.SubElement(source, "object", {"key": "type", "type": "str"}).text = "image"

                if (image := module.find("*[@key='image']")) is not None:
                    image.set("key", "value")
                    source.append(image)
                    module.remove(image)

        if (version := tree.find("*[@key='version']")) is not None:
            version.text = str(self.version)

        ET.indent(tree)
        tree.write(data_path, "utf-8")

        return True


class _(Upgrader):
    version = 30

    def upgrade(self, path: Path) -> bool:
        data_path = path / "project" / "data.xml"

        if not data_path.exists():
            return False

        tree = ET.ElementTree(file = data_path)

        if tree.findtext("*[@key='version']", "0") != str(self.previous_version):
            return False

        if (module_order := tree.find("*[@key='session']/*[@key='pipeline']/*[@key='module_order']")) is not None:
            ET.SubElement(module_order, "object", {"type": "str"}).text = "random_sampling"

        if (modules := tree.find("*[@key='session']/*[@key='pipeline']/*[@key='modules']")) is not None:
            random_sampling = ET.SubElement(modules, "object", {"key": "random_sampling", "type": "temporal.pipeline_modules.RandomSamplingModule"})
            ET.SubElement(random_sampling, "object", {"key": "enabled", "type": "bool"}).text = "False"
            ET.SubElement(random_sampling, "object", {"key": "chance", "type": "float"}).text = "1.0"
            ET.SubElement(random_sampling, "object", {"key": "buffer", "type": "NoneType"})

        if (version := tree.find("*[@key='version']")) is not None:
            version.text = str(self.version)

        ET.indent(tree)
        tree.write(data_path, "utf-8")

        return True


class _(Upgrader):
    version = 31

    def upgrade(self, path: Path) -> bool:
        def create_after(parent: ET.Element, sibling: ET.Element, tag: str, attrs: dict[str, str], text: str) -> ET.Element:
            elem = ET.Element(tag, attrs)
            elem.text = text
            parent.insert(list(parent).index(sibling) + 1, elem)
            return elem

        def create_module(modules: ET.Element, key: str, type: str, enabled: bool, plot_every_nth_frame: int) -> None:
            elem = ET.SubElement(modules, "object", {"key": key, "type": type})
            ET.SubElement(elem, "object", {"key": "enabled", "type": "bool"}).text = str(enabled)
            ET.SubElement(elem, "object", {"key": "plot_every_nth_frame", "type": "int"}).text = str(plot_every_nth_frame)
            ET.SubElement(elem, "object", {"key": "data", "type": "NoneType"})
            ET.SubElement(elem, "object", {"key": "count", "type": "int"}).text = "0"

        data_path = path / "project" / "data.xml"

        if not data_path.exists():
            return False

        tree = ET.ElementTree(file = data_path)

        if tree.findtext("*[@key='version']", "0") != str(self.previous_version):
            return False

        if (module_order := tree.find("*[@key='session']/*[@key='pipeline']/*[@key='module_order']")) is not None:
            if (measuring := module_order.find("*[.='measuring']")) is not None:
                elem = measuring

                elem = create_after(module_order, elem, "object", {"type": "str"}, "color_level_mean_measuring")
                elem = create_after(module_order, elem, "object", {"type": "str"}, "color_level_sigma_measuring")
                elem = create_after(module_order, elem, "object", {"type": "str"}, "luminance_mean_measuring")
                elem = create_after(module_order, elem, "object", {"type": "str"}, "luminance_sigma_measuring")
                elem = create_after(module_order, elem, "object", {"type": "str"}, "noise_sigma_measuring")

                module_order.remove(measuring)

        if (modules := tree.find("*[@key='session']/*[@key='pipeline']/*[@key='modules']")) is not None:
            if (measuring := modules.find("*[@key='measuring']")) is not None:
                enabled = measuring.findtext("*[@key='enabled']", "False") == "True"
                plot_every_nth_frame = int(measuring.findtext("*[@key='plot_every_nth_frame']", "10"))

                create_module(modules, "color_level_mean_measuring", "temporal.measuring_modules.ColorLevelMeanMeasuringModule", enabled, plot_every_nth_frame)
                create_module(modules, "color_level_sigma_measuring", "temporal.measuring_modules.ColorLevelSigmaMeasuringModule", enabled, plot_every_nth_frame)
                create_module(modules, "luminance_mean_measuring", "temporal.measuring_modules.LuminanceMeanMeasuringModule", enabled, plot_every_nth_frame)
                create_module(modules, "luminance_sigma_measuring", "temporal.measuring_modules.LuminanceSigmaMeasuringModule", enabled, plot_every_nth_frame)
                create_module(modules, "noise_sigma_measuring", "temporal.measuring_modules.NoiseSigmaMeasuringModule", enabled, plot_every_nth_frame)

                modules.remove(measuring)

        if (version := tree.find("*[@key='version']")) is not None:
            version.text = str(self.version)

        ET.indent(tree)
        tree.write(data_path, "utf-8")

        return True


class _(Upgrader):
    version = 32

    def upgrade(self, path: Path) -> bool:
        def update_unit_field(unit: ET.Element, key: str) -> None:
            if (field := unit.find(f"*[@key='instance.{key}']")) is not None:
                field.set("key", key)

        data_path = path / "project" / "data.xml"

        if not data_path.exists():
            return False

        tree = ET.ElementTree(file = data_path)

        if tree.findtext("*[@key='version']", "0") != str(self.previous_version):
            return False

        if (controlnet_units := tree.find("*[@key='session']/*[@key='controlnet_units']")) is not None:
            if controlnet_units.get("type", "NoneType") != "NoneType":
                controlnet_units.set("type", "temporal.interop.ControlNetUnitList")

                for unit in controlnet_units:
                    update_unit_field(unit, "enabled")
                    update_unit_field(unit, "module")
                    update_unit_field(unit, "model")
                    update_unit_field(unit, "weight")
                    update_unit_field(unit, "image")
                    update_unit_field(unit, "resize_mode")
                    update_unit_field(unit, "low_vram")
                    update_unit_field(unit, "processor_res")
                    update_unit_field(unit, "threshold_a")
                    update_unit_field(unit, "threshold_b")
                    update_unit_field(unit, "guidance_start")
                    update_unit_field(unit, "guidance_end")
                    update_unit_field(unit, "pixel_perfect")
                    update_unit_field(unit, "control_mode")

                    ET.SubElement(unit, "object", {"key": "effective_region_mask", "type": "NoneType"})

        if (version := tree.find("*[@key='version']")) is not None:
            version.text = str(self.version)

        ET.indent(tree)
        tree.write(data_path, "utf-8")

        return True


class _(Upgrader):
    version = 33

    def upgrade(self, path: Path) -> bool:
        data_path = path / "project" / "data.xml"

        if not data_path.exists():
            return False

        tree = ET.ElementTree(file = data_path)

        if tree.findtext("*[@key='version']", "0") != str(self.previous_version):
            return False

        if (module_order := tree.find("*[@key='session']/*[@key='pipeline']/*[@key='module_order']")) is not None:
            ET.SubElement(module_order, "object", {"type": "str"}).text = "pixelization"

        if (modules := tree.find("*[@key='session']/*[@key='pipeline']/*[@key='modules']")) is not None:
            pixelization = ET.SubElement(modules, "object", {"key": "pixelization", "type": "temporal.image_filters.PixelizationFilter"})
            ET.SubElement(pixelization, "object", {"key": "enabled", "type": "bool"}).text = "False"
            ET.SubElement(pixelization, "object", {"key": "amount", "type": "float"}).text = "1.0"
            ET.SubElement(pixelization, "object", {"key": "amount_relative", "type": "bool"}).text = "False"
            ET.SubElement(pixelization, "object", {"key": "blend_mode", "type": "str"}).text = "normal"
            mask = ET.SubElement(pixelization, "object", {"key": "mask", "type": "temporal.image_mask.ImageMask"})
            ET.SubElement(mask, "object", {"key": "image", "type": "NoneType"})
            ET.SubElement(mask, "object", {"key": "normalized", "type": "bool"}).text = "False"
            ET.SubElement(mask, "object", {"key": "inverted", "type": "bool"}).text = "False"
            ET.SubElement(mask, "object", {"key": "blurring", "type": "float"}).text = "0.0"
            ET.SubElement(pixelization, "object", {"key": "pixel_size", "type": "int"}).text = "1"

        if (version := tree.find("*[@key='version']")) is not None:
            version.text = str(self.version)

        ET.indent(tree)
        tree.write(data_path, "utf-8")

        return True


class _(Upgrader):
    version = 34

    def upgrade(self, path: Path) -> bool:
        data_path = path / "project" / "data.xml"

        if not data_path.exists():
            return False

        tree = ET.ElementTree(file = data_path)

        if tree.findtext("*[@key='version']", "0") != str(self.previous_version):
            return False

        if (pipeline := tree.find("*[@key='session']/*[@key='pipeline']")) is not None:
            actual_module_order: list[str] = []

            if (module_order := pipeline.find("*[@key='module_order']")) is not None:
                for elem in module_order:
                    actual_module_order.append(elem.text or "")

                pipeline.remove(module_order)

            if (modules := pipeline.find("*[@key='modules']")) is not None:
                detached_modules: dict[str, ET.Element] = {}

                for elem in list(modules):
                    detached_modules[elem.get("key", "")] = elem
                    modules.remove(elem)

                for key in actual_module_order:
                    modules.append(detached_modules[key])

        if (version := tree.find("*[@key='version']")) is not None:
            version.text = str(self.version)

        ET.indent(tree)
        tree.write(data_path, "utf-8")

        return True


class _(Upgrader):
    version = 35

    def upgrade(self, path: Path) -> bool:
        data_path = path / "project" / "data.xml"

        if not data_path.exists():
            return False

        tree = ET.ElementTree(file = data_path)

        if tree.findtext("*[@key='version']", "0") != str(self.previous_version):
            return False

        project = tree.getroot()

        if (session := project.find("*[@key='session']")) is not None:
            for elem in list(session):
                project.append(elem)
                session.remove(elem)

            project.remove(session)

        if (initial_noise := project.find("*[@key='initial_noise']")) is not None:
            initial_noise.set("type", "temporal.project.InitialNoiseParams")

        if (iteration := project.find("*[@key='iteration']")) is not None:
            iteration.set("type", "temporal.project.IterationData")

        if (version := tree.find("*[@key='version']")) is not None:
            version.text = str(self.version)

        ET.indent(tree)
        tree.write(data_path, "utf-8")

        return True


class _(Upgrader):
    version = 36

    def upgrade(self, path: Path) -> bool:
        data_path = path / "project" / "data.xml"

        if not data_path.exists():
            return False

        tree = ET.ElementTree(file = data_path)

        if tree.findtext("*[@key='version']", "0") != str(self.previous_version):
            return False

        if (initial_noise := tree.find("*[@key='initial_noise']")) is not None:
            if (use_initial_seed := initial_noise.find("*[@key='use_initial_seed']")) is not None:
                if (noise := initial_noise.find("*[@key='noise']")) is not None:
                    ET.SubElement(noise, "object", {"key": "use_global_seed", "type": "bool"}).text = use_initial_seed.text

                initial_noise.remove(use_initial_seed)

        if (noise_overlay := tree.find("*[@key='pipeline']/*[@key='modules']/*[@key='noise_overlay']")) is not None:
            if (use_dynamic_seed := noise_overlay.find("*[@key='use_dynamic_seed']")) is not None:
                if (noise := noise_overlay.find("*[@key='noise']")) is not None:
                    ET.SubElement(noise, "object", {"key": "use_global_seed", "type": "bool"}).text = use_dynamic_seed.text

                noise_overlay.remove(use_dynamic_seed)

        if (version := tree.find("*[@key='version']")) is not None:
            version.text = str(self.version)

        ET.indent(tree)
        tree.write(data_path, "utf-8")

        return True


class _(Upgrader):
    version = 37

    def upgrade(self, path: Path) -> bool:
        data_path = path / "project" / "data.xml"

        if not data_path.exists():
            return False

        tree = ET.ElementTree(file = data_path)

        if tree.findtext("*[@key='version']", "0") != str(self.previous_version):
            return False

        for base_path in (
            "*[@key='initial_noise']",
            "*[@key='pipeline']/*[@key='modules']/*[@key='noise_overlay']",
        ):
            if (noise := tree.find(f"{base_path}/*[@key='noise']")) is not None:
                ET.SubElement(noise, "object", {"key": "mode", "type": "str"}).text = "fbm"

                if (octaves := noise.find("*[@key='octaves']")) is not None:
                    octaves.set("key", "detail")
                    octaves.set("type", "float")

        if (version := tree.find("*[@key='version']")) is not None:
            version.text = str(self.version)

        ET.indent(tree)
        tree.write(data_path, "utf-8")

        return True


class _(Upgrader):
    version = 38

    def upgrade(self, path: Path) -> bool:
        data_path = path / "project" / "data.xml"

        if not data_path.exists():
            return False

        tree = ET.ElementTree(file = data_path)

        if tree.findtext("*[@key='version']", "0") != str(self.previous_version):
            return False

        if (modules := tree.find("*[@key='pipeline']/*[@key='modules']")) is not None:
            gradient_overlay = ET.SubElement(modules, "object", {"key": "gradient_overlay", "type": "temporal.image_filters.GradientOverlayFilter"})
            ET.SubElement(gradient_overlay, "object", {"key": "enabled", "type": "bool"}).text = "False"
            ET.SubElement(gradient_overlay, "object", {"key": "amount", "type": "float"}).text = "1.0"
            ET.SubElement(gradient_overlay, "object", {"key": "amount_relative", "type": "bool"}).text = "False"
            ET.SubElement(gradient_overlay, "object", {"key": "blend_mode", "type": "str"}).text = "normal"

            mask = ET.SubElement(gradient_overlay, "object", {"key": "mask", "type": "temporal.image_mask.ImageMask"})
            ET.SubElement(mask, "object", {"key": "image", "type": "NoneType"})
            ET.SubElement(mask, "object", {"key": "normalized", "type": "bool"}).text = "False"
            ET.SubElement(mask, "object", {"key": "inverted", "type": "bool"}).text = "False"
            ET.SubElement(mask, "object", {"key": "blurring", "type": "float"}).text = "0.0"

            gradient = ET.SubElement(gradient_overlay, "object", {"key": "gradient", "type": "temporal.gradient.Gradient"})
            ET.SubElement(gradient, "object", {"key": "type", "type": "str"}).text = "linear"
            ET.SubElement(gradient, "object", {"key": "start_x", "type": "float"}).text = "0.0"
            ET.SubElement(gradient, "object", {"key": "start_y", "type": "float"}).text = "0.0"
            ET.SubElement(gradient, "object", {"key": "end_x", "type": "float"}).text = "1.0"
            ET.SubElement(gradient, "object", {"key": "end_y", "type": "float"}).text = "1.0"

            start_color = ET.SubElement(gradient, "object", {"key": "start_color", "type": "temporal.color.Color"})
            ET.SubElement(start_color, "object", {"key": "r", "type": "float"}).text = "1.0"
            ET.SubElement(start_color, "object", {"key": "g", "type": "float"}).text = "1.0"
            ET.SubElement(start_color, "object", {"key": "b", "type": "float"}).text = "1.0"
            ET.SubElement(start_color, "object", {"key": "a", "type": "float"}).text = "1.0"

            end_color = ET.SubElement(gradient, "object", {"key": "end_color", "type": "temporal.color.Color"})
            ET.SubElement(end_color, "object", {"key": "r", "type": "float"}).text = "0.0"
            ET.SubElement(end_color, "object", {"key": "g", "type": "float"}).text = "0.0"
            ET.SubElement(end_color, "object", {"key": "b", "type": "float"}).text = "0.0"
            ET.SubElement(end_color, "object", {"key": "a", "type": "float"}).text = "1.0"

        if (version := tree.find("*[@key='version']")) is not None:
            version.text = str(self.version)

        ET.indent(tree)
        tree.write(data_path, "utf-8")

        return True


class _(Upgrader):
    version = 39

    def upgrade(self, path: Path) -> bool:
        data_path = path / "project" / "data.xml"

        if not data_path.exists():
            return False

        tree = ET.ElementTree(file = data_path)

        if tree.findtext("*[@key='version']", "0") != str(self.previous_version):
            return False

        if (modules := tree.find("*[@key='pipeline']/*[@key='modules']")) is not None:
            pattern_overlay = ET.SubElement(modules, "object", {"key": "pattern_overlay", "type": "temporal.image_filters.PatternOverlayFilter"})
            ET.SubElement(pattern_overlay, "object", {"key": "enabled", "type": "bool"}).text = "False"
            ET.SubElement(pattern_overlay, "object", {"key": "amount", "type": "float"}).text = "1.0"
            ET.SubElement(pattern_overlay, "object", {"key": "amount_relative", "type": "bool"}).text = "False"
            ET.SubElement(pattern_overlay, "object", {"key": "blend_mode", "type": "str"}).text = "normal"

            mask = ET.SubElement(pattern_overlay, "object", {"key": "mask", "type": "temporal.image_mask.ImageMask"})
            ET.SubElement(mask, "object", {"key": "image", "type": "NoneType"})
            ET.SubElement(mask, "object", {"key": "normalized", "type": "bool"}).text = "False"
            ET.SubElement(mask, "object", {"key": "inverted", "type": "bool"}).text = "False"
            ET.SubElement(mask, "object", {"key": "blurring", "type": "float"}).text = "0.0"

            pattern = ET.SubElement(pattern_overlay, "object", {"key": "pattern", "type": "temporal.pattern.Pattern"})
            ET.SubElement(pattern, "object", {"key": "type", "type": "str"}).text = "horizontal_lines"
            ET.SubElement(pattern, "object", {"key": "size", "type": "int"}).text = "8"

            color_a = ET.SubElement(pattern, "object", {"key": "color_a", "type": "temporal.color.Color"})
            ET.SubElement(color_a, "object", {"key": "r", "type": "float"}).text = "1.0"
            ET.SubElement(color_a, "object", {"key": "g", "type": "float"}).text = "1.0"
            ET.SubElement(color_a, "object", {"key": "b", "type": "float"}).text = "1.0"
            ET.SubElement(color_a, "object", {"key": "a", "type": "float"}).text = "1.0"

            color_b = ET.SubElement(pattern, "object", {"key": "color_b", "type": "temporal.color.Color"})
            ET.SubElement(color_b, "object", {"key": "r", "type": "float"}).text = "0.0"
            ET.SubElement(color_b, "object", {"key": "g", "type": "float"}).text = "0.0"
            ET.SubElement(color_b, "object", {"key": "b", "type": "float"}).text = "0.0"
            ET.SubElement(color_b, "object", {"key": "a", "type": "float"}).text = "1.0"

        if (version := tree.find("*[@key='version']")) is not None:
            version.text = str(self.version)

        ET.indent(tree)
        tree.write(data_path, "utf-8")

        return True


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


class _(Upgrader):
    version = 41

    def upgrade(self, path: Path) -> bool:
        data_path = path / "project" / "data.xml"

        if not data_path.exists():
            return False

        tree = ET.ElementTree(file = data_path)

        if tree.findtext("*[@key='version']", "0") != str(self.previous_version):
            return False

        classes = {
            "normal": "NormalBlendMode",
            "add": "AddBlendMode",
            "subtract": "SubtractBlendMode",
            "multiply": "MultiplyBlendMode",
            "divide": "DivideBlendMode",
            "lighten": "LightenBlendMode",
            "darken": "DarkenBlendMode",
            "hard_light": "HardLightBlendMode",
            "soft_light": "SoftLightBlendMode",
            "color_dodge": "ColorDodgeBlendMode",
            "color_burn": "ColorBurnBlendMode",
            "overlay": "OverlayBlendMode",
            "screen": "ScreenBlendMode",
            "difference": "DifferenceBlendMode",
            "exclusion": "ExclusionBlendMode",
            "hue": "HueBlendMode",
            "saturation": "SaturationBlendMode",
            "value": "ValueBlendMode",
            "color": "ColorBlendMode",
        }

        for blend_mode in tree.iterfind("*[@key='pipeline']/*[@key='modules']/*/*[@key='blend_mode']"):
            if blend_mode.text is not None:
                blend_mode.set("type", f"temporal.blend_modes.{classes[blend_mode.text]}")
                blend_mode.text = None

        if (version := tree.find("*[@key='version']")) is not None:
            version.text = str(self.version)

        ET.indent(tree)
        tree.write(data_path, "utf-8")

        return True


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


class _(Upgrader):
    version = 43

    def upgrade(self, path: Path) -> bool:
        data_path = path / "project" / "data.xml"

        if not data_path.exists():
            return False

        tree = ET.ElementTree(file = data_path)

        if tree.findtext("*[@key='version']", "0") != str(self.previous_version):
            return False

        if (modules := tree.find("*[@key='pipeline']/*[@key='modules']")) is not None:
            modules.append(ET.fromstring("""
            <object type="temporal.pipeline_modules.filtering.sepia.SepiaFilter">
              <object key="enabled" type="bool">False</object>
              <object key="amount" type="float">1.0</object>
              <object key="amount_relative" type="bool">False</object>
              <object key="blend_mode" type="temporal.blend_modes.NormalBlendMode" />
              <object key="mask" type="temporal.image_mask.ImageMask">
                <object key="image" type="NoneType" />
                <object key="normalized" type="bool">False</object>
                <object key="inverted" type="bool">False</object>
                <object key="blurring" type="float">0.0</object>
              </object>
            </object>
            """))

        if (version := tree.find("*[@key='version']")) is not None:
            version.text = str(self.version)

        ET.indent(tree)
        tree.write(data_path, "utf-8")

        return True


class _(Upgrader):
    version = 44

    def upgrade(self, path: Path) -> bool:
        data_path = path / "project" / "data.xml"

        if not data_path.exists():
            return False

        tree = ET.ElementTree(file = data_path)

        if tree.findtext("*[@key='version']", "0") != str(self.previous_version):
            return False

        if (modules := tree.find("*[@key='pipeline']/*[@key='modules']")) is not None:
            modules.append(ET.fromstring("""
            <object type="temporal.pipeline_modules.filtering.color_matrix.ColorMatrixFilter">
              <object key="enabled" type="bool">False</object>
              <object key="amount" type="float">1.0</object>
              <object key="amount_relative" type="bool">False</object>
              <object key="blend_mode" type="temporal.blend_modes.NormalBlendMode" />
              <object key="mask" type="temporal.image_mask.ImageMask">
                <object key="image" type="NoneType" />
                <object key="normalized" type="bool">False</object>
                <object key="inverted" type="bool">False</object>
                <object key="blurring" type="float">0.0</object>
              </object>
              <object key="r" type="temporal.color.Color">
                <object key="r" type="float">1.0</object>
                <object key="g" type="float">0.0</object>
                <object key="b" type="float">0.0</object>
                <object key="a" type="float">1.0</object>
              </object>
              <object key="g" type="temporal.color.Color">
                <object key="r" type="float">0.0</object>
                <object key="g" type="float">1.0</object>
                <object key="b" type="float">0.0</object>
                <object key="a" type="float">1.0</object>
              </object>
              <object key="b" type="temporal.color.Color">
                <object key="r" type="float">0.0</object>
                <object key="g" type="float">0.0</object>
                <object key="b" type="float">1.0</object>
                <object key="a" type="float">1.0</object>
              </object>
              <object key="normalized" type="bool">False</object>
            </object>
            """))

        if (version := tree.find("*[@key='version']")) is not None:
            version.text = str(self.version)

        ET.indent(tree)
        tree.write(data_path, "utf-8")

        return True


class _(Upgrader):
    version = 45

    def upgrade(self, path: Path) -> bool:
        data_path = path / "project" / "data.xml"

        if not data_path.exists():
            return False

        tree = ET.ElementTree(file = data_path)

        if tree.findtext("*[@key='version']", "0") != str(self.previous_version):
            return False

        if (modules := tree.find("*[@key='pipeline']/*[@key='modules']")) is not None:
            modules.append(ET.fromstring("""
            <object type="temporal.pipeline_modules.filtering.displacement.DisplacementFilter">
              <object key="enabled" type="bool">False</object>
              <object key="amount" type="float">1.0</object>
              <object key="amount_relative" type="bool">False</object>
              <object key="blend_mode" type="temporal.blend_modes.NormalBlendMode" />
              <object key="mask" type="temporal.image_mask.ImageMask">
                <object key="image" type="NoneType" />
                <object key="normalized" type="bool">False</object>
                <object key="inverted" type="bool">False</object>
                <object key="blurring" type="float">0.0</object>
              </object>
              <object key="source" type="temporal.image_source.ImageSource">
                <object key="type" type="str">image</object>
                <object key="value" type="NoneType" />
              </object>
              <object key="x_scale" type="float">0.0</object>
              <object key="y_scale" type="float">0.0</object>
            </object>
            """))

        if (version := tree.find("*[@key='version']")) is not None:
            version.text = str(self.version)

        ET.indent(tree)
        tree.write(data_path, "utf-8")

        return True


class _(Upgrader):
    version = 46

    def upgrade(self, path: Path) -> bool:
        data_path = path / "project" / "data.xml"

        if not data_path.exists():
            return False

        tree = ET.ElementTree(file = data_path)

        if tree.findtext("*[@key='version']", "0") != str(self.previous_version):
            return False

        if (module := tree.find("*[@key='pipeline']/*[@key='modules']/*[@type='temporal.pipeline_modules.temporal.random_sampling.RandomSamplingModule']")) is not None:
            ET.SubElement(module, "object", {"key": "opacity", "type": "float"}).text = "1.0"

        if (version := tree.find("*[@key='version']")) is not None:
            version.text = str(self.version)

        ET.indent(tree)
        tree.write(data_path, "utf-8")

        return True


class _(Upgrader):
    version = 47

    def upgrade(self, path: Path) -> bool:
        data_path = path / "project" / "data.xml"

        if not data_path.exists():
            return False

        tree = ET.ElementTree(file = data_path)

        if tree.findtext("*[@key='version']", "0") != str(self.previous_version):
            return False

        if (modules := tree.find("*[@key='pipeline']/*[@key='modules']")) is not None:
            modules.append(ET.fromstring("""
            <object type="temporal.pipeline_modules.measuring.difference.DifferenceMeasuringModule">
              <object key="enabled" type="bool">False</object>
              <object key="plot_every_nth_frame" type="int">10</object>
              <object key="data" type="NoneType" />
              <object key="count" type="int">0</object>
              <object key="last_images" type="list" />
            </object>
            """))

        if (version := tree.find("*[@key='version']")) is not None:
            version.text = str(self.version)

        ET.indent(tree)
        tree.write(data_path, "utf-8")

        return True


class _(Upgrader):
    version = 48

    def upgrade(self, path: Path) -> bool:
        data_path = path / "project" / "data.xml"

        if not data_path.exists():
            return False

        tree = ET.ElementTree(file = data_path)

        if tree.findtext("*[@key='version']", "0") != str(self.previous_version):
            return False

        if (modules := tree.find("*[@key='pipeline']/*[@key='modules']")) is not None:
            modules.append(ET.fromstring("""
            <object type="temporal.pipeline_modules.filtering.tiling.TilingFilter">
              <object key="enabled" type="bool">False</object>
              <object key="amount" type="float">1.0</object>
              <object key="amount_relative" type="bool">False</object>
              <object key="blend_mode" type="temporal.blend_modes.NormalBlendMode" />
              <object key="mask" type="temporal.image_mask.ImageMask">
                <object key="image" type="NoneType" />
                <object key="normalized" type="bool">False</object>
                <object key="inverted" type="bool">False</object>
                <object key="blurring" type="float">0.0</object>
              </object>
              <object key="exponent" type="float">1.0</object>
            </object>
            """))

        if (version := tree.find("*[@key='version']")) is not None:
            version.text = str(self.version)

        ET.indent(tree)
        tree.write(data_path, "utf-8")

        return True
