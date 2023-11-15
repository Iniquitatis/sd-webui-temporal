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
