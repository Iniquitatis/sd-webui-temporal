import xml.etree.ElementTree as ET
from pathlib import Path

from temporal.compat.upgrader import Upgrader


class _(Upgrader):
    version = 50

    def upgrade(self, path: Path) -> bool:
        data_path = path / "project" / "data.xml"

        if not data_path.exists():
            return False

        tree = ET.ElementTree(file = data_path)

        if tree.findtext("*[@key='version']", "0") != str(self.previous_version):
            return False

        root = tree.getroot()

        sampler_mapping = {
            "DDIM":                          ("DDIM",              "Automatic"),
            "DPM adaptive":                  ("DPM adaptive",      "Automatic"),
            "DPM fast":                      ("DPM fast",          "Automatic"),
            "DPM++ 2M":                      ("DPM++ 2M",          "Automatic"),
            "DPM++ 2M Karras":               ("DPM++ 2M",          "Karras"),
            "DPM++ 2M SDE":                  ("DPM++ 2M SDE",      "Automatic"),
            "DPM++ 2M SDE Exponential":      ("DPM++ 2M SDE",      "Exponential"),
            "DPM++ 2M SDE Karras":           ("DPM++ 2M SDE",      "Karras"),
            "DPM++ 2M SDE Heun":             ("DPM++ 2M SDE Heun", "Automatic"),
            "DPM++ 2M SDE Heun Exponential": ("DPM++ 2M SDE Heun", "Exponential"),
            "DPM++ 2M SDE Heun Karras":      ("DPM++ 2M SDE Heun", "Karras"),
            "DPM++ 2S a":                    ("DPM++ 2S a",        "Automatic"),
            "DPM++ 2S a Karras":             ("DPM++ 2S a",        "Karras"),
            "DPM++ 3M SDE":                  ("DPM++ 3M SDE",      "Automatic"),
            "DPM++ 3M SDE Exponential":      ("DPM++ 3M SDE",      "Exponential"),
            "DPM++ 3M SDE Karras":           ("DPM++ 3M SDE",      "Karras"),
            "DPM++ SDE":                     ("DPM++ SDE",         "Automatic"),
            "DPM++ SDE Karras":              ("DPM++ SDE",         "Karras"),
            "DPM2":                          ("DPM2",              "Automatic"),
            "DPM2 Karras":                   ("DPM2",              "Karras"),
            "DPM2 a":                        ("DPM2 a",            "Automatic"),
            "DPM2 a Karras":                 ("DPM2 a",            "Karras"),
            "Euler":                         ("Euler",             "Automatic"),
            "Euler a":                       ("Euler a",           "Automatic"),
            "Heun":                          ("Heun",              "Automatic"),
            "LCM":                           ("LCM",               "Automatic"),
            "LMS":                           ("LMS",               "Automatic"),
            "LMS Karras":                    ("LMS",               "Karras"),
            "PLMS":                          ("PLMS",              "Automatic"),
            "Restart":                       ("Restart",           "Automatic"),
            "UniPC":                         ("UniPC",             "Automatic"),
        }

        if ((processing := root.find("*[@key='processing']")) is not None and
            (sampler_name := processing.find("*[@key='sampler_name']")) is not None and
            (processing.find("*[@key='scheduler']")) is None):
            old_sampler_text = sampler_name.text or ""

            sampler_text, scheduler_text = sampler_mapping.get(old_sampler_text, ("", "Automatic"))

            if not sampler_text:
                for new_sampler_text, _ in sorted(sampler_mapping.values(), key = lambda x: 1e9 - len(x[0])):
                    if old_sampler_text.startswith(new_sampler_text):
                        sampler_text = new_sampler_text
                        break
                else:
                    sampler_text = "Euler a"

            sampler_name.text = sampler_text
            ET.SubElement(processing, "object", {"key": "scheduler", "type": "str"}).text = scheduler_text

        parameters = ET.SubElement(root, "object", {"key": "parameters", "type": "temporal.backends.webui.WebUIImageToImageParams"})
        ET.SubElement(parameters, "object", {"key": "model", "type": "str"}).text = root.findtext("*[@key='options']/*[@key='sd_model_checkpoint']", "")
        ET.SubElement(parameters, "object", {"key": "vae", "type": "str"}).text = root.findtext("*[@key='options']/*[@key='sd_vae']", "")
        ET.SubElement(parameters, "object", {"key": "clip_skip", "type": "int"}).text = root.findtext("*[@key='options']/*[@key='CLIP_stop_at_last_layers']", "1")
        positive_prompts = ET.SubElement(parameters, "object", {"key": "positive_prompts", "type": "list"})
        ET.SubElement(positive_prompts, "object", {"type": "str"}).text = root.findtext("*[@key='processing']/*[@key='prompt']", "")
        negative_prompts = ET.SubElement(parameters, "object", {"key": "negative_prompts", "type": "list"})
        ET.SubElement(negative_prompts, "object", {"type": "str"}).text = root.findtext("*[@key='processing']/*[@key='negative_prompt']", "")
        ET.SubElement(parameters, "object", {"key": "width", "type": "int"}).text = root.findtext("*[@key='processing']/*[@key='width']", "512")
        ET.SubElement(parameters, "object", {"key": "height", "type": "int"}).text = root.findtext("*[@key='processing']/*[@key='height']", "512")
        ET.SubElement(parameters, "object", {"key": "sampler", "type": "str"}).text = root.findtext("*[@key='processing']/*[@key='sampler_name']", "Euler a")
        ET.SubElement(parameters, "object", {"key": "scheduler", "type": "str"}).text = root.findtext("*[@key='processing']/*[@key='scheduler']", "Automatic")
        ET.SubElement(parameters, "object", {"key": "steps", "type": "int"}).text = root.findtext("*[@key='processing']/*[@key='steps']", "20")
        ET.SubElement(parameters, "object", {"key": "cfg", "type": "float"}).text = root.findtext("*[@key='processing']/*[@key='cfg_scale']", "5.0")
        ET.SubElement(parameters, "object", {"key": "strength", "type": "float"}).text = root.findtext("*[@key='processing']/*[@key='denoising_strength']", "0.5")
        seeds = ET.SubElement(parameters, "object", {"key": "seeds", "type": "list"})
        ET.SubElement(seeds, "object", {"type": "int"}).text = root.findtext("*[@key='processing']/*[@key='seed']", "0")
        images = ET.SubElement(parameters, "object", {"key": "images", "type": "list"})
        ET.SubElement(images, "object", {"type": "PIL.Image.Image"}).text = root.findtext("*[@key='processing']/*[@key='init_images']/*[1]", "")

        if (options := root.find("*[@key='options']")) is not None:
            root.remove(options)
            parameters.append(options)

        if (processing := root.find("*[@key='processing']")) is not None:
            root.remove(processing)
            parameters.append(processing)

        if (controlnet_units := root.find("*[@key='controlnet_units']")) is not None:
            root.remove(controlnet_units)
            parameters.append(controlnet_units)

            if controlnet_units.get("type", "None") != "None":
                controlnet_units.set("type", "temporal.backends.webui.controlnet.ControlNetUnitList")

                for unit in controlnet_units:
                    unit.set("type", "temporal.backends.webui.controlnet.ControlNetUnitWrapper")

        if (version := tree.find("*[@key='version']")) is not None:
            version.text = str(self.version)

        ET.indent(tree)
        tree.write(data_path, "utf-8")

        return True
