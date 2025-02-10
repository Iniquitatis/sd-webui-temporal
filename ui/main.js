import {Accordion} from "./scripts/base/accordion.js";
import {Button} from "./scripts/base/button.js";
import {Checkbox} from "./scripts/base/checkbox.js";
import {CodeArea} from "./scripts/base/code_area.js";
import {Column} from "./scripts/base/column.js";
import {ImageBox} from "./scripts/base/image_box.js";
import {MultiStateButton} from "./scripts/base/multi_state_button.js";
import {NumberBox} from "./scripts/base/number_box.js";
import {ProgressBar} from "./scripts/base/progress_bar.js";
import {Tabs} from "./scripts/base/tabs.js";
import {TextArea} from "./scripts/base/text_area.js";
import {TextBox} from "./scripts/base/text_box.js";
import {getRequest, postRequest} from "./scripts/utils/requests.js";
import {FSStoreBox} from "./scripts/fs_store_box.js";
import {InitialNoiseEditor} from "./scripts/initial_noise_editor.js";
import {PipelineEditor} from "./scripts/pipeline_editor.js";
import {ProcessingParamsEditor} from "./scripts/processing_params_editor.js";
import {blendModes, models, pipelineModules, presets, projects, samplers, schedulers, vaes} from "./scripts/test_data.js";
import {VideoRendererEditor} from "./scripts/video_renderer_editor.js";

export class MainUI extends Column {
    constructor() {
        super();

        let project = {};

        this.createChild(MultiStateButton, (e) => {
            e.style.height = "calc(var(--widget-height) * 2)";
            e.onStateChange.connect((state) => {
                if (state == "active") {
                    this._progressBar.value = 0;
                    this._progressBar.style.display = null;

                    this._progressInterval = window.setInterval(async () => {
                        await getRequest("/temporal/state")
                        .then((result) => {
                            if (result.state == "active") {
                                this._progressBar.value = result.current_iteration;
                                this._progressBar.total = result.total_iterations;
                                this._progressBar.text = `${this._progressBar.value} / ${this._progressBar.total}`;
                            } else if (result.state == "stopped") {
                                e.state = "stopped";
                            }
                        });

                        await getRequest("/temporal/preview")
                        .then((result) => {
                            if (!result) return;

                            this._preview.value = `data:image/png;base64,${result}`;
                        });
                    }, 1000);

                    postRequest("/temporal/generate", project);
                } else if (state == "stopped") {
                    this._progressBar.style.display = "none";

                    window.clearInterval(this._progressInterval);
                    this._progressInterval = null;

                    postRequest("/temporal/interrupt", {});
                }
            });
        }, {"stopped": "Generate", "active": "Stop"});

        this._progressBar = this.createChild(ProgressBar, (e) => {
            e.total = 100;
            e.style.display = "none";
        });

        this._progressInterval = null;

        this._preview = this.createChild(ImageBox, (e) => {
            e.label = "Preview";
            e.height = "30rem";
        });

        this.createChild(FSStoreBox, (e) => {
            e.label = "Preset";
            e.entries = Object.keys(presets);
        }, "presets", ["refresh", "load", "save", "rename", "delete"]);

        this.createChild(FSStoreBox, (e) => {
            e.label = "Project";
            e.entries = Object.keys(projects);
        }, "projects", ["refresh", "load", "rename", "delete"]);

        this.createChild(Tabs, (e) => {
            e.createTab("General", Column, (e) => {
                e.createChild(TextBox, (e) => {
                    e.label = "Name";
                });

                e.createChild(TextArea, (e) => {
                    e.label = "Description";
                });

                e.createChild(Checkbox, (e) => {
                    e.label = "Load parameters";
                    e.value = true;
                });

                e.createChild(Checkbox, (e) => {
                    e.label = "Continue from last frame";
                    e.value = true;
                });

                e.createChild(NumberBox, (e) => {
                    e.label = "Iteration count";
                    e.minimum = 1;
                    e.step = 1;
                    e.value = 10;
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
                        project.pipeline = value;
                    });
                });

                e.createChild(CodeArea, (e) => {
                    e.label = "Animation",
                    e.onValueChange.connect((value) => {
                        project.animation = value;
                    });
                });
            });

            e.createTab("Video Rendering", Column, (e) => {
                e.createChild(VideoRendererEditor);
            });

            e.createTab("Measuring", Column, (e) => {
                e.createChild(NumberBox, (e) => {
                    e.label = "Parallel index";
                    e.minimum = 1;
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
    });

    await getRequest("/temporal/models")
    .then((result) => {
        for (let model of result) {
            models[model] = model;
        }
    });

    await getRequest("/temporal/pipeline_modules")
    .then((result) => {
        for (let [k, v] of Object.entries(result)) {
            pipelineModules[k] = v;
        }
    });

    await getRequest("/temporal/presets")
    .then((result) => {
        for (let preset of result) {
            presets[preset] = preset;
        }
    });

    await getRequest("/temporal/projects")
    .then((result) => {
        for (let project of result) {
            projects[project] = project;
        }
    });

    await getRequest("/temporal/samplers")
    .then((result) => {
        for (let sampler of result) {
            samplers[sampler] = sampler;
        }
    });

    await getRequest("/temporal/schedulers")
    .then((result) => {
        for (let scheduler of result) {
            schedulers[scheduler] = scheduler;
        }
    });

    await getRequest("/temporal/vaes")
    .then((result) => {
        for (let vae of result) {
            vaes[vae] = vae;
        }
    });

    document.body.appendChild(new MainUI());
};
