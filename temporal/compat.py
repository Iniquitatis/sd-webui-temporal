from pathlib import Path
from shutil import copy2
from typing import Any

import numpy as np

from temporal.utils.fs import ensure_directory_exists, load_json, load_text, move_entry, save_json, save_text
from temporal.utils.func import make_func_registerer
from temporal.utils.image import load_image, pil_to_np

VERSION = 15
UPGRADERS, upgrader = make_func_registerer()

def upgrade_project(path: Path) -> None:
    last_version = 0

    for version, upgrader in UPGRADERS.items():
        if upgrader.func(path):
            last_version = version

    if last_version:
        print(f"Upgraded project to version {last_version}")

@upgrader(1)
def _(path: Path) -> bool:
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
    save_text(version_path, "1")

    return True

@upgrader(2)
def _(path: Path) -> bool:
    version_path = path / "session" / "version.txt"
    params_path = path / "session" / "parameters.json"

    if int(load_text(version_path, "0")) != 1:
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
    save_text(version_path, "2")

    return True

@upgrader(3)
def _(path: Path) -> bool:
    version_path = path / "session" / "version.txt"

    if int(load_text(version_path, "0")) != 2:
        return False

    if frames := sorted(path.glob("*.png"), key = lambda x: int(x.stem)):
        copy2(frames[-1], ensure_directory_exists(path / "session" / "buffer") / "001.png")

    save_text(version_path, "3")

    return True

@upgrader(4)
def _(path: Path) -> bool:
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

    if int(load_text(version_path, "0")) != 3:
        return False

    data = load_json(params_path, {})

    data["shared_params"] = upgrade_values(data.get("shared_params", {}))
    data["generation_params"] = upgrade_values(data.get("generation_params", {}))

    for i, unit_data in enumerate(data.get("controlnet_params", [])):
        data["controlnet_params"][i] = upgrade_values(unit_data)

    data["extension_params"] = upgrade_values(data.get("extension_params", {}))

    save_json(params_path, data)
    save_text(version_path, "4")

    return True

@upgrader(5)
def _(path: Path) -> bool:
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

    if int(load_text(version_path, "0")) != 4:
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

    save_text(version_path, "5")

    return True

@upgrader(6)
def _(path: Path) -> bool:
    version_path = path / "session" / "version.txt"
    params_path = path / "session" / "parameters.json"

    if int(load_text(version_path, "0")) != 5:
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
    save_text(version_path, "6")

    return True

@upgrader(7)
def _(path: Path) -> bool:
    version_path = path / "session" / "version.txt"
    params_path = path / "session" / "parameters.json"

    if int(load_text(version_path, "0")) != 6:
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
    save_text(version_path, "7")

    return True

@upgrader(8)
def _(path: Path) -> bool:
    version_path = path / "session" / "version.txt"
    params_path = path / "session" / "parameters.json"

    if int(load_text(version_path, "0")) != 7:
        return False

    data = load_json(params_path, {})

    ext_params = data["extension_params"]

    for key in ["multisampling_algorithm", "frame_merging_algorithm"]:
        if ext_params[key] == "mean":
            ext_params[key] = "arithmetic_mean"

    save_json(params_path, data)
    save_text(version_path, "8")

    return True

@upgrader(9)
def _(path: Path) -> bool:
    version_path = path / "session" / "version.txt"
    params_path = path / "session" / "parameters.json"

    if int(load_text(version_path, "0")) != 8:
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
    save_text(version_path, "9")

    return True

@upgrader(10)
def _(path: Path) -> bool:
    version_path = path / "session" / "version.txt"
    params_path = path / "session" / "parameters.json"

    if int(load_text(version_path, "0")) != 9:
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
    save_text(version_path, "10")

    return True

@upgrader(11)
def _(path: Path) -> bool:
    version_path = path / "session" / "version.txt"
    params_path = path / "session" / "parameters.json"

    if int(load_text(version_path, "0")) != 10:
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
    save_text(version_path, "11")

    return True

@upgrader(12)
def _(path: Path) -> bool:
    version_path = path / "session" / "version.txt"
    params_path = path / "session" / "parameters.json"

    if int(load_text(version_path, "0")) != 11:
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
    save_text(version_path, "12")

    return True

@upgrader(13)
def _(path: Path) -> bool:
    version_path = path / "session" / "version.txt"
    params_path = path / "session" / "parameters.json"

    if int(load_text(version_path, "0")) != 12:
        return False

    data = load_json(params_path, {})

    ext_params = data["extension_params"]
    ext_params.update({
        "symmetry_horizontal": True,
        "symmetry_vertical": False,
    })

    save_json(params_path, data)
    save_text(version_path, "13")

    return True

@upgrader(14)
def _(path: Path) -> bool:
    version_path = path / "session" / "version.txt"
    params_path = path / "session" / "parameters.json"

    if int(load_text(version_path, "0")) != 13:
        return False

    data = load_json(params_path, {})

    ext_params = data["extension_params"]
    ext_params.update({
        "median_percentile": 50.0,
    })

    save_json(params_path, data)
    save_text(version_path, "14")

    return True

@upgrader(15)
def _(path: Path) -> bool:
    version_path = path / "session" / "version.txt"

    if int(load_text(version_path, "0")) != 14:
        return False

    project_data_dir = ensure_directory_exists(path / "project")
    move_entry(path / "metrics", project_data_dir / "metrics")
    move_entry(path / "session" / "buffer", project_data_dir / "buffer")
    move_entry(path / "session", project_data_dir / "session")
    move_entry(version_path, project_data_dir / "version.txt")

    save_text(project_data_dir / "version.txt", "15")

    return True
