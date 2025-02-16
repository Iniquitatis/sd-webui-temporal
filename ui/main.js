import {Accordion} from "./scripts/base/accordion.js";
import {Button} from "./scripts/base/button.js";
import {Checkbox} from "./scripts/base/checkbox.js";
import {CodeArea} from "./scripts/base/code_area.js";
import {Column} from "./scripts/base/column.js";
import {Form} from "./scripts/base/form.js";
import {ImageBox} from "./scripts/base/image_box.js";
import {MultiStateButton} from "./scripts/base/multi_state_button.js";
import {NumberBox} from "./scripts/base/number_box.js";
import {ProgressBar} from "./scripts/base/progress_bar.js";
import {Row} from "./scripts/base/row.js";
import {Tabs} from "./scripts/base/tabs.js";
import {TextArea} from "./scripts/base/text_area.js";
import {TextBox} from "./scripts/base/text_box.js";
import {VideoBox} from "./scripts/base/video_box.js";
import {mapObject} from "../scripts/utils/object.js";
import {getRequest, postRequest} from "./scripts/utils/requests.js";
import {FSStoreBox} from "./scripts/fs_store_box.js";
import {InitialNoiseEditor} from "./scripts/initial_noise_editor.js";
import {ModuleList} from "./scripts/module_list.js";
import {PipelineModuleEditor} from "./scripts/pipeline_module_editor.js";
import {ProcessingParamsEditor} from "./scripts/processing_params_editor.js";
import {blendModes, models, pipelineModules, presets, projects, samplers, schedulers, vaes, videoFilters} from "./scripts/shared_data.js";
import {VideoRendererEditor} from "./scripts/video_renderer_editor.js";

export class MainUI extends Column {
    constructor() {
        super();

        let generation = {
            project: {
                general: {},
            },
        };
        let preset = {
            project: {
                general: {},
            },
        };

        this.createChild(MultiStateButton, (e) => {
            e.states = {stopped: "Generate", active: "Stop"};
            e.state = "stopped";
            e.style.height = "calc(var(--widget-height) * 2)";
            e.onStateChange.connect(async (state) => {
                if (state == "active") {
                    this._progressBar.value = 0;
                    this._progressBar.total = 0;
                    this._progressBar.text = "(Indeterminate)";
                    this._progressBar.style.display = null;

                    this._progressInterval = window.setInterval(async () => {
                        let state = await getRequest("/temporal/state");

                        if (state.state == "active") {
                            this._progressBar.value = state.current_iteration;
                            this._progressBar.total = state.total_iterations;
                            this._progressBar.text = `${this._progressBar.value} / ${this._progressBar.total}`;
                        } else if (state.state == "stopped") {
                            e.state = "stopped";
                        }

                        let preview = await getRequest("/temporal/preview");

                        if (!preview) return;

                        this._preview.value = `data:image/png;base64,${preview}`;
                    }, 1000);

                    await postRequest("/temporal/generate", generation);
                } else if (state == "stopped") {
                    this._progressBar.style.display = "none";

                    window.clearInterval(this._progressInterval);
                    this._progressInterval = null;

                    await postRequest("/temporal/interrupt");
                }
            });
        });

        this._progressBar = this.createChild(ProgressBar, (e) => {
            e.style.display = "none";
        });

        this._progressInterval = null;

        this._preview = this.createChild(ImageBox, (e) => {
            e.height = "30rem";
        });

        this.createChild(Form, (e) => {
            e.createField("Preset", FSStoreBox, (e) => {
                e.entries = Object.keys(presets);
                e.saveCallback = () => preset;
                e.onLoad.connect((value) => {
                    // FIXME: Unsure where exactly it should be fixed
                    value.project.general.parameters.images = value.project.general.parameters.images.map((data) => `data:image/png;base64,${data}`);

                    this._name.value = value.name;
                    this._loadParameters.value = value.load_parameters;
                    this._continueFromLastFrame.value = value.continue_from_last_frame;
                    this._iterCount.value = value.iter_count;
                    this._processing.value = value.project.general.parameters;
                    this._initialNoise.value = value.project.initial_noise;
                    this._parallel.value = value.project.general.parallel;
                    this._pipelineModules.value = value.project.modules;
                    this._animation.value = value.project.animation.code;
                    this._videoRenderer.value = value.video_renderer;
                    this._measuringParallelIndex.value = value.measuring_parallel_index;
                });
            }, "presets", ["refresh", "load", "save", "rename", "delete"]);

            e.createField("Project", FSStoreBox, (e) => {
                e.entries = Object.keys(projects);
                e.onValueChange.connect((value) => {
                    this._name.value = value;
                });
                e.onLoad.connect((value) => {
                    // FIXME: Unsure where exactly it should be fixed
                    value.general.parameters.images = value.general.parameters.images.map((data) => `data:image/png;base64,${data}`);

                    this._processing.value = value.general.parameters;
                    this._initialNoise.value = value.initial_noise;
                    this._parallel.value = value.general.parallel;
                    this._pipelineModules.value = value.modules;
                });
            }, "projects", ["refresh", "load", "rename", "delete"]);
        });

        this.createChild(Tabs, (e) => {
            e.createTab("General", Form, (e) => {
                this._name = e.createField("Name", TextBox, (e) => {
                    e.onValueChange.connect((value) => {
                        generation.name = value;
                        preset.name = value;
                    });
                });

                e.createField("Description", TextArea);

                this._loadParameters = e.createField("Load parameters", Checkbox, (e) => {
                    e.value = true;
                    e.onValueChange.connect((value) => {
                        generation.load_parameters = value;
                        preset.load_parameters = value;
                    });
                });

                this._continueFromLastFrame = e.createField("Continue from last frame", Checkbox, (e) => {
                    e.value = true;
                    e.onValueChange.connect((value) => {
                        generation.continue_from_last_frame = value;
                        preset.continue_from_last_frame = value;
                    });
                });

                this._iterCount = e.createField("Iteration count", NumberBox, (e) => {
                    e.minimum = 1;
                    e.step = 1;
                    e.value = 10;
                    e.onValueChange.connect((value) => {
                        generation.iter_count = value;
                        preset.iter_count = value;
                    });
                });
            });

            this._processing = e.createTab("Processing", ProcessingParamsEditor, (e) => {
                e.onValueChange.connect((value) => {
                    generation.project.general.parameters = value;
                    preset.project.general.parameters = value;
                });
                // FIXME: Temporary
                e.onValueChange.connect((value) => console.log(value));
            });

            e.createTab("Pipeline", Column, (e) => {
                e.createChild(Accordion, (e) => {
                    e.label = "Initial noise";

                    this._initialNoise = e.createChild(InitialNoiseEditor, (e) => {
                        e.onValueChange.connect((value) => {
                            generation.project.initial_noise = value;
                            preset.project.initial_noise = value;
                        });
                        // FIXME: Temporary
                        e.onValueChange.connect((value) => console.log(value));
                    });
                });

                e.createChild(Form, (e) => {
                    this._parallel = e.createField("Parallel", NumberBox, (e) => {
                        e.minimum = 1;
                        e.step = 1;
                        e.value = 1;
                        e.onValueChange.connect((value) => {
                            generation.project.general.parallel = value;
                            preset.project.general.parallel = value;
                        });
                    });

                    this._pipelineModules = e.createField("Add module", ModuleList, (e) => {
                        e.onValueChange.connect((value) => {
                            generation.project.modules = value;
                            preset.project.modules = value;
                        });
                        // FIXME: Temporary
                        e.onValueChange.connect((value) => console.log(value));
                    }, PipelineModuleEditor, mapObject(pipelineModules, (_, module) => `${module.icon} ${module.name}`), pipelineModules);

                    this._animation = e.createField("Animation", CodeArea, (e) => {
                        e.onValueChange.connect((value) => {
                            generation.project.animation = {code: value};
                            preset.project.animation = {code: value};
                        });
                    });
                });
            });

            e.createTab("Video Rendering", Form, (e) => {
                this._videoRenderer = e.createChild(VideoRendererEditor, (e) => {
                    e.onValueChange.connect((value) => {
                        preset.video_renderer = value;
                    });
                    // FIXME: Temporary
                    e.onValueChange.connect((value) => console.log(value));
                });

                this._videoParallelIndex = e.createField("Parallel index", NumberBox, (e) => {
                    e.minimum = 1;
                    e.step = 1;
                    e.value = 1;
                    e.onValueChange.connect((value) => {
                        preset.video_parallel_index = value;
                    });
                    // FIXME: Temporary
                    e.onValueChange.connect((value) => console.log(value));
                });

                this._videoRenderButtonRow = e.createChild(Row, (e) => {
                    for (let type of ["draft", "final"]) {
                        e.createChild(Button, (e) => {
                            e.label = `Render ${type}`;
                            e.style.width = "100%";
                            e.onClick.connect(async () => {
                                for (let button of this._videoRenderButtonRow.childNodes) {
                                    button.classList.add("disabled");
                                }

                                let data = await postRequest("/temporal/render_video", {
                                    "type": type,
                                    "data": this._videoRenderer.value,
                                    "parallel_index": this._videoParallelIndex.value,
                                });

                                if (!data) return;

                                this._videoPreview.value = `data:video/mp4;base64,${data}`;

                                for (let button of this._videoRenderButtonRow.childNodes) {
                                    button.classList.remove("disabled");
                                }
                            });
                        });
                    }
                });

                this._videoPreview = e.createChild(VideoBox);
            });

            e.createTab("Measuring", Form, (e) => {
                this._measuringParallelIndex = e.createField("Parallel index", NumberBox, (e) => {
                    e.minimum = 1;
                    e.step = 1;
                    e.value = 1;
                    e.onValueChange.connect((value) => {
                        preset.measuring_parallel_index = value;
                    });
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

// FIXME: Just a temporary crutch, as those collections should be both objects
// and arrays, not just objects
function createMappingFromArray(array) {
    let result = {};

    for (let value of array) {
        result[value] = value;
    }

    return result;
}

window.onload = async () => {
    Object.assign(blendModes, await getRequest("/temporal/blend_modes"));
    Object.assign(models, createMappingFromArray(await getRequest("/temporal/models")));
    Object.assign(pipelineModules, await getRequest("/temporal/pipeline_modules"));
    Object.assign(presets, createMappingFromArray(await getRequest("/temporal/presets")));
    Object.assign(projects, createMappingFromArray(await getRequest("/temporal/projects")));
    Object.assign(samplers, createMappingFromArray(await getRequest("/temporal/samplers")));
    Object.assign(schedulers, createMappingFromArray(await getRequest("/temporal/schedulers")));
    Object.assign(vaes, createMappingFromArray(await getRequest("/temporal/vaes")));
    Object.assign(videoFilters, await getRequest("/temporal/video_filters"));

    document.body.appendChild(new MainUI());
};
