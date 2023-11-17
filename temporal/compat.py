import json

UPGRADERS = dict()

def upgrade_project(path):
    for version, func in UPGRADERS.items():
        if not func(path):
            print(f"WARNING: Couldn't upgrade project to version {version}")
            return False

    return True

def upgrader(version):
    def decorator(func):
        UPGRADERS[version] = func
        return func
    return decorator

@upgrader(1)
def _(path):
    def upgrade_value(value):
        if isinstance(value, list):
            return {"type": "list", "data": [upgrade_value(x) for x in value]}
        elif isinstance(value, dict):
            if "im_type" in value:
                return {"type": value["im_type"], "filename": value["filename"]}
            else:
                return {"type": "dict", "data": {k: upgrade_value(v) for k, v in value.items()}}
        else:
            return value

    def upgrade_values(d):
        return {k: upgrade_value(v) for k, v in d.items()}

    if (version_path := (path / "session" / "version.txt")).is_file():
        return True

    if not (params_path := (path / "session" / "parameters.json")).is_file():
        return False

    with open(params_path, "r", encoding = "utf-8") as file:
        data = json.load(file)

    data["shared_params"] = upgrade_values(data.get("shared_params", {}))
    data["generation_params"] = upgrade_values(data.get("generation_params", {}))

    for i, unit_data in enumerate(data.get("controlnet_params", [])):
        data["controlnet_params"][i] = upgrade_values(unit_data)

    data["extension_params"] = upgrade_values(data.get("extension_params", {}))

    with open(params_path, "w", encoding = "utf-8") as file:
        json.dump(data, file, indent = 4)

    with open(version_path, "w") as file:
        file.write("1")

    return True

@upgrader(2)
def _(path):
    if not (version_path := (path / "session" / "version.txt")).is_file():
        return False

    with open(version_path, "r") as file:
        if int(file.read()) >= 2:
            return True

    if not (params_path := (path / "session" / "parameters.json")).is_file():
        return False

    with open(params_path, "r", encoding = "utf-8") as file:
        data = json.load(file)

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

    with open(params_path, "w", encoding = "utf-8") as file:
        json.dump(data, file, indent = 4)

    with open(version_path, "w") as file:
        file.write("2")

    return True
