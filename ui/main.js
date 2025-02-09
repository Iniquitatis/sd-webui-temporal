import {Accordion} from "./scripts/base/accordion.js";
import {BoolEditor} from "./scripts/base/bool_editor.js";
import {Button} from "./scripts/base/button.js";
import {Column} from "./scripts/base/column.js";
import {EnumEditor} from "./scripts/base/enum_editor.js";
import {ImageEditor} from "./scripts/base/image_editor.js";
import {NumberEditor} from "./scripts/base/number_editor.js";
import {Row} from "./scripts/base/row.js";
import {Tabs} from "./scripts/base/tabs.js";
import {TextEditor} from "./scripts/base/text_editor.js";
import {ToolButton} from "./scripts/base/tool_button.js";
import {createElement} from "./scripts/utils/dom.js";
import {getRequest, postRequest} from "./scripts/utils/requests.js";
import {InitialNoiseEditor} from "./scripts/initial_noise_editor.js";
import {PipelineEditor} from "./scripts/pipeline_editor.js";
import {ProcessingParamsEditor} from "./scripts/processing_params_editor.js";
import {blendModes, models, pipelineModules, presets, projects, samplers, schedulers, vaes} from "./scripts/test_data.js";
import {VideoRendererEditor} from "./scripts/video_renderer_editor.js";

export class MainUI extends Column {
    constructor() {
        super();

        let project = {};

        this.createChild(Button, (e) => {
            e.label = "Generate";
            e.style.height = "calc(var(--widget-height) * 2)";
            e.onClick.connect(() => {
                if (e.label == "Generate") {
                    this._progressBar.value = 0;
                    this._progressBar.style.display = null;

                    this._progressInterval = window.setInterval(() => {
                        this._progressBar.value += 1;
                        this._progressBar.value %= 100;

                        getRequest("/temporal/preview")
                        .then((result) => {
                            if (!result) return;

                            this._preview.value = `data:image/png;base64,${result}`;
                        });
                    }, 1000);

                    postRequest("/temporal/generate", {
                        "parameters": project.parameters,
                        "modules": project.modules,
                        "iter_count": project.iter_count,
                    })
                    .then(() => {
                        // TODO: Stop process here
                    });

                    e.label = "Stop";
                } else {
                    this._progressBar.style.display = "none";

                    window.clearInterval(this._progressInterval);
                    this._progressInterval = null;

                    postRequest("/temporal/interrupt", {});

                    e.label = "Generate";
                }
            });
        });

        this._progressInterval = null;

        this._progressBar = createElement(this, "progress", (e) => {
            e.max = 100;
            e.value = 0;
            e.style.display = "none";
            e.style.height = "2rem";
            e.style.marginTop = "calc(var(--layout-gap) * -1)";
            e.style.width = "100%";
        });

        this._preview = this.createChild(ImageEditor, (e) => {
            e.label = "Preview";
            e.height = "30rem";
        });

        this.createChild(EnumEditor, (e) => {
            e.label = "Preset";
            e.variant = "menu";
            e.choices = presets;

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
                    e.onValueChange.connect((value) => {
                        project.iter_count = value;
                    });
                });
            });

            e.createTab("Processing", Column, (e) => {
                e.createChild(ProcessingParamsEditor, (e) => {
                    e.onValueChange.connect((value) => {
                        project.parameters = value;
                    });
                });
            });

            e.createTab("Pipeline", Column, (e) => {
                e.createChild(Accordion, (e) => {
                    e.label = "Initial noise";

                    e.createChild(InitialNoiseEditor, (e) => {
                        e.onValueChange.connect((value) => {
                            project.initial_noise = value;
                        });
                    });
                });

                e.createChild(PipelineEditor, (e) => {
                    e.onValueChange.connect((value) => {
                        project.modules = value.modules;
                    });
                });
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
    await getRequest("/temporal/blend_modes")
    .then((result) => {
        for (let [k, v] of Object.entries(result)) {
            blendModes[k] = v;
        }
    })
    .catch(() => {});

    await getRequest("/temporal/models")
    .then((result) => {
        for (let model of result) {
            models[model] = model;
        }
    })
    .catch(() => {});

    await getRequest("/temporal/pipeline_modules")
    .then((result) => {
        for (let [k, v] of Object.entries(result)) {
            pipelineModules[k] = v;
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

    await getRequest("/temporal/samplers")
    .then((result) => {
        for (let sampler of result) {
            samplers[sampler] = sampler;
        }
    })
    .catch(() => {});

    await getRequest("/temporal/schedulers")
    .then((result) => {
        for (let scheduler of result) {
            schedulers[scheduler] = scheduler;
        }
    })
    .catch(() => {});

    await getRequest("/temporal/vaes")
    .then((result) => {
        for (let vae of result) {
            vaes[vae] = vae;
        }
    })
    .catch(() => {});

    document.body.appendChild(new MainUI());
};
