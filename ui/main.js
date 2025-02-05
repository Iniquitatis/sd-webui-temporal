import {Button, ToolButton} from "./scripts/core/buttons.js";
import {Column, Row, Tabs} from "./scripts/core/layout.js";
import {BoolEditor, EnumEditor, ImageEditor, NumberEditor, TextEditor} from "./scripts/core/value_editors.js";
import {getObjectKeyByIndex} from "./scripts/utils/object.js";
import {getRequest, postRequest} from "./scripts/utils/requests.js";
import {PipelineEditor} from "./scripts/pipeline_editor.js";
import {VideoRendererEditor} from "./scripts/video_renderer_editor.js";
import {blendModes, modules, presets, projects} from "./scripts/test_data.js";

export class MainUI extends Column {
    constructor() {
        super();

        this.createChild(Button, (e) => {
            e.label = "Generate";
            e._button.style.lineHeight = "calc(var(--line-height) * 2)";
            e.onClick.connect(() => {
                if (e.label == "Generate") {
                    this._progressBar.value = 0;
                    this._progressBar.style.display = null;

                    this._progressInterval = window.setInterval(() => {
                        this._progressBar.value += 1;
                        this._progressBar.value %= 100;
                    }, 1000);

                    postRequest("/sdapi/v1/txt2img", {
                        "prompt": "female, happy, dynamic, colorful, smooth, soft, 3d render",
                        "negative_prompt": "",
                        "seed": -1,
                        "sampler_name": "DPM++ 3M SDE",
                        "scheduler": "DDIM",
                        "batch_size": 4,
                        "n_iter": 1,
                        "steps": 10,
                        "cfg_scale": 2.5,
                        "width": 768,
                        "height": 1152,
                    })
                    .then((result) => {
                        this._preview._image.innerHTML = "";

                        for (let data of result.images) {
                            let img = document.createElement("img");
                            img.src = `data:image/png;base64,${data}`;
                            this._preview._image.appendChild(img);
                        }
                    });

                    e.label = "Stop";
                } else {
                    this._progressBar.style.display = "none";

                    window.clearInterval(this._progressInterval);
                    this._progressInterval = null;

                    postRequest("/sdapi/v1/interrupt", {});

                    e.label = "Generate";
                }
            });
        });

        this._progressInterval = null;

        this._progressBar = document.createElement("progress");
        this._progressBar.max = 100;
        this._progressBar.value = 0;
        this._progressBar.style.display = "none";
        this._progressBar.style.height = "2rem";
        this._progressBar.style.marginTop = "calc(var(--layout-gap) * -1)";
        this._progressBar.style.width = "100%";
        this.appendChild(this._progressBar);

        this._preview = this.createChild(ImageEditor, (e) => {
            e.label = "Preview";
            e._input.style.height = "50vh";
        });

        this.createChild(EnumEditor, (e) => {
            e.label = "Preset";
            e.variant = "menu";
            e.choices = presets;
            e.value = getObjectKeyByIndex(e.choices, 0);

            e.createChild(Row, (e) => {
                e.createChild(ToolButton, (e) => {
                    e.label = "\u{0001f504}";
                });

                e.createChild(ToolButton, (e) => {
                    e.label = "\u{0001f4c2}";
                });

                e.createChild(ToolButton, (e) => {
                    e.label = "\u{0001f4be}";
                });

                e.createChild(ToolButton, (e) => {
                    e.label = "\u{270f}\u{fe0f}";
                });

                e.createChild(ToolButton, (e) => {
                    e.label = "\u{0001f5d1}\u{fe0f}";
                });
            });
        });

        this.createChild(EnumEditor, (e) => {
            e.label = "Project";
            e.variant = "menu";
            e.choices = projects;
            e.value = getObjectKeyByIndex(e.choices, 0);

            e.createChild(Row, (e) => {
                e.createChild(ToolButton, (e) => {
                    e.label = "\u{0001f504}";
                });

                e.createChild(ToolButton, (e) => {
                    e.label = "\u{0001f4c2}";
                });

                e.createChild(ToolButton, (e) => {
                    e.label = "\u{270f}\u{fe0f}";
                });

                e.createChild(ToolButton, (e) => {
                    e.label = "\u{0001f5d1}\u{fe0f}";
                });
            });
        });

        this.createChild(Tabs, (e) => {
            e.createTab("General", Column, (e) => {
                e.createChild(TextEditor, (e) => {
                    e.label = "Name";
                    e.variant = "box";
                });

                e.createChild(TextEditor, (e) => {
                    e.label = "Description";
                    e.variant = "area";
                });

                e.createChild(Row, (e) => {
                    e.createChild(NumberEditor, (e) => {
                        e.label = "Width";
                        e.variant = "slider";
                        e.minimum = 64;
                        e.maximum = 2048;
                        e.step = 8;
                        e.value = 512;
                    });

                    e.createChild(NumberEditor, (e) => {
                        e.label = "Height";
                        e.variant = "slider";
                        e.minimum = 64;
                        e.maximum = 2048;
                        e.step = 8;
                        e.value = 512;
                    });
                });

                e.createChild(BoolEditor, (e) => {
                    e.label = "Load parameters";
                    e.value = true;
                });

                e.createChild(BoolEditor, (e) => {
                    e.label = "Continue from last frame";
                    e.value = true;
                });

                e.createChild(NumberEditor, (e) => {
                    e.label = "Iteration count";
                    e.variant = "box";
                    e.minimum = 1;
                    e.maximum = 2 ** 32 - 1;
                    e.step = 1;
                    e.value = 100;
                });
            });

            e.createTab("Pipeline", Column, (e) => {
                e.createChild(PipelineEditor);
            });

            e.createTab("Video Rendering", Column, (e) => {
                e.createChild(VideoRendererEditor);
            });

            e.createTab("Measuring", Column, (e) => {
                e.createChild(NumberEditor, (e) => {
                    e.label = "Parallel index";
                    e.variant = "box";
                    e.minimum = 1;
                    e.maximum = 2 ** 32 - 1;
                    e.step = 1;
                    e.value = 1;
                });

                e.createChild(Button, (e) => {
                    e.label = "Render graphs";
                });
            });

            e.createTab("Tools", Column, (e) => {
                e.createChild(Button, (e) => {
                    e.label = "Delete intermediate frames";
                });

                e.createChild(Button, (e) => {
                    e.label = "Delete session data";
                });
            });

            e.createTab("Settings", Column, (e) => {
                e.createChild(Button, (e) => {
                    e.label = "Apply";
                });
            });

            e.createTab("Help", Column, (e) => {
                e.innerText = "Blah";
            });
        });
    }
}
customElements.define("main-ui", MainUI);

window.onload = async () => {
    await getRequest("/temporal/pipeline_modules")
    .then((result) => {
        for (let [k, v] of Object.entries(result)) {
            modules[k] = v;
        }
    })
    .catch(() => {});

    await getRequest("/temporal/blend_modes")
    .then((result) => {
        for (let [k, v] of Object.entries(result)) {
            blendModes[k] = v;
        }
    })
    .catch(() => {});

    await getRequest("/temporal/presets")
    .then((result) => {
        for (let preset of result) {
            presets[preset] = preset;
        }
    })
    .catch(() => {});

    await getRequest("/temporal/projects")
    .then((result) => {
        for (let project of result) {
            projects[project] = project;
        }
    })
    .catch(() => {});

    document.body.appendChild(new MainUI());
};
