from modules.shared import opts

from temporal.compat import upgrade_project
from temporal.fs import load_json, recreate_directory, save_json, save_text
from temporal.image_preprocessing import iterate_all_preprocessor_keys
from temporal.image_utils import load_image
from temporal.interop import import_cn
from temporal.serialization import load_dict, load_object, save_dict, save_object

def get_last_frame_index(frame_dir):
    def get_index(path):
        if path.is_file():
            try:
                return int(path.stem)
            except:
                print(f"WARNING: {path} doesn't match the frame name format")

        return 0

    return max((get_index(path) for path in frame_dir.glob("*.png")), default = 0)

def load_last_frame(frame_dir):
    if index := get_last_frame_index(frame_dir):
        return load_image(frame_dir / f"{index:05d}.png")

    return None

def load_session(p, ext_params, project_dir):
    if not (session_dir := (project_dir / "session")).is_dir():
        return

    if not (params_path := (session_dir / "parameters.json")).is_file():
        return

    if not upgrade_project(project_dir):
        return

    data = load_json(params_path, {})

    # NOTE: `p.override_settings` juggles VAEs back-and-forth, slowing down the process considerably
    load_dict(opts.data, data.get("shared_params", {}), session_dir)

    load_object(p, data.get("generation_params", {}), session_dir)

    if external_code := import_cn():
        for unit_data, cn_unit in zip(data.get("controlnet_params", []), external_code.get_all_units_in_processing(p)):
            load_object(cn_unit, unit_data, session_dir)

    load_object(ext_params, data.get("extension_params", {}), session_dir)

def save_session(p, ext_params, project_dir):
    session_dir = recreate_directory(project_dir / "session")

    save_json(session_dir / "parameters.json", dict(
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
            "image_mask",
            "resize_mode",
            "mask_blur_x",
            "mask_blur_y",
            "inpainting_mask_invert",
            "inpainting_fill",
            "inpaint_full_res",
            "inpaint_full_res_padding",
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
            "noise_for_first_frame",
            "use_sd",
            "multisampling_samples",
            "multisampling_batch_size",
            "multisampling_trimming",
            "multisampling_easing",
            "multisampling_preference",
            "frame_merging_frames",
            "frame_merging_trimming",
            "frame_merging_easing",
            "frame_merging_preference",
            "preprocessing_order",
        ] + list(iterate_all_preprocessor_keys())),
    ))
    save_text(session_dir / "version.txt", "9")
