import json

from modules.shared import opts

from temporal.compat import upgrade_project
from temporal.image_preprocessing import iterate_all_preprocessor_keys
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

def load_session(p, ext_params, project_dir, session_dir, last_index):
    if not (params_path := (session_dir / "parameters.json")).is_file():
        return

    if not upgrade_project(project_dir):
        return

    with open(params_path, "r", encoding = "utf-8") as params_file:
        data = json.load(params_file)

    # NOTE: `p.override_settings` juggles VAEs back-and-forth, slowing down the process considerably
    load_dict(opts.data, data.get("shared_params", {}), session_dir)

    load_object(p, data.get("generation_params", {}), session_dir)

    if external_code := import_cn():
        for unit_data, cn_unit in zip(data.get("controlnet_params", []), external_code.get_all_units_in_processing(p)):
            load_object(cn_unit, unit_data, session_dir)

    load_object(ext_params, data.get("extension_params", {}), session_dir)

    if (im_path := (project_dir / f"{last_index:05d}.png")).is_file():
        p.init_images = [load_image(im_path)]

    # FIXME: Unreliable; works properly only on first generation, but not on the subsequent ones
    if p.seed != -1:
        p.seed = p.seed + last_index

def save_session(p, ext_params, project_dir, session_dir, last_index):
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
        extension_params = save_object(ext_params, session_dir, [
            "save_every_nth_frame",
            "archive_mode",
        ] + list(iterate_all_preprocessor_keys())),
    )

    with open(session_dir / "parameters.json", "w", encoding = "utf-8") as params_file:
        json.dump(data, params_file, indent = 4)

    with open(session_dir / "version.txt", "w") as version_file:
        version_file.write("2")
