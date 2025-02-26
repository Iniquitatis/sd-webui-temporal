import {Button} from "./scripts/base/button.js";
import {Checkbox} from "./scripts/base/checkbox.js";
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
import {getRequest, postRequest} from "./scripts/utils/requests.js";
import {FSStoreBox} from "./scripts/fs_store_box.js";
import {GeneralDataEditor} from "./scripts/general_data_editor.js";
import {OptionsEditor} from "./scripts/options_editor.js";
import {PipelineEditor} from "./scripts/pipeline_editor.js";
import {blendModes, models, optionCategories, pipelineModules, presets, projects, samplers, schedulers, vaes, videoFilters} from "./scripts/shared_data.js";
import {VideoRendererEditor} from "./scripts/video_renderer_editor.js";

export class MainUI extends Column {
    constructor() {
        super();

        let generation = {
            project: {
                general: {},
                pipeline: {},
            },
        };
        let preset = {
            project: {
                general: {},
                pipeline: {},
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

                        this._preview.value = preview;
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
                    this._name.value = value.name;
                    this._general.value = value.project.general;
                    this._loadParameters.value = value.load_parameters;
                    this._continueFromLastFrame.value = value.continue_from_last_frame;
                    this._iterCount.value = value.iter_count;
                    this._pipeline.value = value.project.pipeline;
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
                    this._general.value = value.general;
                    this._pipeline.value = value.pipeline;
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

                this._general = e.createChild(GeneralDataEditor, (e) => {
                    e.onValueChange.connect((value) => {
                        generation.project.general = value;
                        preset.project.general = value;
                    });
                    // FIXME: Temporary
                    e.onValueChange.connect((value) => console.log(value));
                });

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

            this._pipeline = e.createTab("Pipeline", PipelineEditor, (e) => {
                e.onValueChange.connect((value) => {
                    generation.project.pipeline = value;
                    preset.project.pipeline = value;
                });
                // FIXME: Temporary
                e.onValueChange.connect((value) => console.log(value));
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

                                this._videoPreview.value = data;

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

            e.createTab("Settings", OptionsEditor, (e) => {
                e.onApply.connect(async (value) => {
                    await postRequest("/temporal/apply_settings", {
                        "data": value,
                    });
                });

                // FIXME: Temporary
                e.onValueChange.connect((value) => console.log(value));
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
    Object.assign(optionCategories, await getRequest("/temporal/option_categories"));
    Object.assign(pipelineModules, await getRequest("/temporal/pipeline_modules"));
    Object.assign(presets, createMappingFromArray(await getRequest("/temporal/presets")));
    Object.assign(projects, createMappingFromArray(await getRequest("/temporal/projects")));
    Object.assign(samplers, createMappingFromArray(await getRequest("/temporal/samplers")));
    Object.assign(schedulers, createMappingFromArray(await getRequest("/temporal/schedulers")));
    Object.assign(vaes, createMappingFromArray(await getRequest("/temporal/vaes")));
    Object.assign(videoFilters, await getRequest("/temporal/video_filters"));

    document.body.appendChild(new MainUI());
};
