import json

from modules.shared import opts

from temporal.image_utils import load_image
from temporal.interop import import_cn
from temporal.serialization import load_dict, load_object, save_dict, save_object

def get_last_frame_index(frame_dir):
    def get_index(path):
        try:
            if path.is_file():
                return int(path.stem)
        except:
            print(f"WARNING: {path} doesn't match the frame name format")
            return 0

    return max((get_index(path) for path in frame_dir.glob("*.png")), default = 0)

def load_session(p, uv, project_dir, session_dir, last_index):
    if not (params_path := (session_dir / "parameters.json")).is_file():
        return

    with open(params_path, "r", encoding = "utf-8") as params_file:
        data = json.load(params_file)

    # NOTE: `p.override_settings` juggles VAEs back-and-forth, slowing down the process considerably
    load_dict(opts.data, data.get("shared_params", {}), session_dir)

    load_object(p, data.get("generation_params", {}), session_dir)

    if external_code := import_cn():
        for unit_data, cn_unit in zip(data.get("controlnet_params", []), external_code.get_all_units_in_processing(p)):
            load_object(cn_unit, unit_data, session_dir)

    load_object(uv, data.get("extension_params", {}), session_dir)

    if (im_path := (project_dir / f"{last_index:05d}.png")).is_file():
        p.init_images = [load_image(im_path)]

    # FIXME: Unreliable; works properly only on first generation, but not on the subsequent ones
    if p.seed != -1:
        p.seed = p.seed + last_index

def save_session(p, uv, project_dir, session_dir, last_index):
    for path in session_dir.glob("*.*"):
        path.unlink()

    data = dict(
        shared_params = save_dict(opts.data, session_dir, [
            "sd_model_checkpoint",
            "sd_vae",
            "CLIP_stop_at_last_layers",
            "always_discard_next_to_last_sigma",
        ]),
        generation_params = save_object(p, session_dir, [
            "prompt",
            "negative_prompt",
            "init_images",
            "resize_mode",
            "sampler_name",
            "steps",
            "refiner_checkpoint",
            "refiner_switch_at",
            "width",
            "height",
            "cfg_scale",
            "denoising_strength",
            "seed",
            "seed_enable_extras",
            "subseed",
            "subseed_strength",
            "seed_resize_from_w",
            "seed_resize_from_h",
        ]),
        controlnet_params = list(
            save_object(cn_unit, session_dir, [
                "image",
                "enabled",
                "low_vram",
                "pixel_perfect",
                "module",
                "model",
                "weight",
                "guidance_start",
                "guidance_end",
                "processor_res",
                "threshold_a",
                "threshold_b",
                "control_mode",
                "resize_mode",
            ])
            for cn_unit in external_code.get_all_units_in_processing(p)
        ) if (external_code := import_cn()) else [],
        extension_params = save_object(uv, session_dir, [
            "save_every_nth_frame",
            "noise_compression_enabled",
            "noise_compression_constant",
            "noise_compression_adaptive",
            "color_correction_enabled",
            "color_correction_image",
            "normalize_contrast",
            "color_balancing_enabled",
            "brightness",
            "contrast",
            "saturation",
            "noise_enabled",
            "noise_amount",
            "noise_relative",
            "noise_mode",
            "modulation_enabled",
            "modulation_amount",
            "modulation_relative",
            "modulation_mode",
            "modulation_image",
            "modulation_blurring",
            "tinting_enabled",
            "tinting_amount",
            "tinting_relative",
            "tinting_mode",
            "tinting_color",
            "sharpening_enabled",
            "sharpening_amount",
            "sharpening_relative",
            "sharpening_radius",
            "transformation_enabled",
            "scaling",
            "rotation",
            "translation_x",
            "translation_y",
            "symmetrize",
            "blurring_enabled",
            "blurring_radius",
            "custom_code_enabled",
            "custom_code",
        ]),
    )

    with open(session_dir / "parameters.json", "w", encoding = "utf-8") as params_file:
        json.dump(data, params_file, indent = 4)
