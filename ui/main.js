import {Button} from "./scripts/base/button.js";
import {CanvasBox} from "./scripts/base/canvas_box.js";
import {Checkbox} from "./scripts/base/checkbox.js";
import {Column} from "./scripts/base/column.js";
import {Form} from "./scripts/base/form.js";
import {MultiStateButton} from "./scripts/base/multi_state_button.js";
import {NumberBox} from "./scripts/base/number_box.js";
import {ProgressBar} from "./scripts/base/progress_bar.js";
import {Row} from "./scripts/base/row.js";
import {Tabs} from "./scripts/base/tabs.js";
import {TextArea} from "./scripts/base/text_area.js";
import {TextBox} from "./scripts/base/text_box.js";
import {VideoBox} from "./scripts/base/video_box.js";
import {FieldManager} from "./scripts/core/field_manager.js";
import {Signal} from "./scripts/core/signal.js";
import {getRequest, postRequest} from "./scripts/utils/requests.js";
import {FSStoreBox} from "./scripts/fs_store_box.js";
import {GeneralDataEditor} from "./scripts/general_data_editor.js";
import {OptionsEditor} from "./scripts/options_editor.js";
import {PipelineEditor} from "./scripts/pipeline_editor.js";
import {blendModes, models, optionCategories, pipelineModules, presets, projects, samplers, schedulers, vaes, videoFilters} from "./scripts/shared_data.js";
import {VectorEditor} from "./scripts/base/vector_editor.js";
import {VideoRendererEditor} from "./scripts/video_renderer_editor.js";

export class MainUI extends Form {
    constructor() {
        super();

        this.onGenerationChange = new Signal();
        this.onPresetChange = new Signal();
        this.onProjectChange = new Signal();

        this._generationManager = new FieldManager(this.onGenerationChange);
        this._presetManager = new FieldManager(this.onPresetChange);
        this._projectManager = new FieldManager(this.onProjectChange);

        this.onProjectChange.connect((value) => {
            {
                let newValue = this._generationManager.value;
                newValue.project = value;

                this._generationManager.value = newValue;
            }

            {
                let newValue = this._presetManager.value;
                newValue.project = value;

                this._presetManager.value = newValue;
            }
        });

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

                    await postRequest("/temporal/generate", this._generationManager.value);
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

        this._preview = this.createChild(CanvasBox, (e) => {
            e.height = "75vh";
            e._element.width = 512;
            e._element.height = 512;
            this._generationManager.manage(e, "image");
            this._presetManager.manage(e, "image");
        });

        this._imageSize = this.createField("Image size", VectorEditor, (e) => {
            e.minimum = 64;
            e.maximum = 2048;
            e.step = 8;
            e.value = {x: 512, y: 512};
            e.onValueChange.connect((value) => {
                this._preview._element.width = value.x;
                this._preview._element.height = value.y;
            });
            this._generationManager.manage(e, "image_size");
            this._presetManager.manage(e, "image_size");
        }, NumberBox, {x: "X", y: "Y"});

        this.createField("Preset", FSStoreBox, (e) => {
            e.entries = Object.keys(presets);
            e.saveCallback = () => this._presetManager.value;
            e.onLoad.connect((value) => {
                this._presetManager.value = value;
                this._projectManager.value = value.project;

                this._preview.value = value.project.general.image;
                this._imageSize.value = value.project.general.image_size;
            });
        }, "presets", ["refresh", "load", "save", "rename", "delete"]);

        this.createField("Project", FSStoreBox, (e) => {
            e.entries = Object.keys(projects);
            e.onValueChange.connect((value) => {
                this._name.value = value;
            });
            e.onLoad.connect((value) => {
                this._projectManager.value = value;

                this._preview.value = value.general.image;
                this._imageSize.value = value.general.image_size;
            });
        }, "projects", ["refresh", "load", "rename", "delete"]);

        this.createChild(Tabs, (e) => {
            e.createTab("General", Form, (e) => {
                this._name = e.createField("Name", TextBox, (e) => {
                    this._generationManager.manage(e, "name");
                    this._presetManager.manage(e, "name");
                });

                e.createField("Description", TextArea);

                e.createChild(GeneralDataEditor, (e) => {
                    this._projectManager.manage(e, "general");
                });

                e.createField("Load parameters", Checkbox, (e) => {
                    e.value = true;
                    this._generationManager.manage(e, "load_parameters");
                    this._presetManager.manage(e, "load_parameters");
                });

                e.createField("Continue from last frame", Checkbox, (e) => {
                    e.value = true;
                    this._generationManager.manage(e, "continue_from_last_frame");
                    this._presetManager.manage(e, "continue_from_last_frame");
                });

                e.createField("Iteration count", NumberBox, (e) => {
                    e.minimum = 1;
                    e.step = 1;
                    e.value = 10;
                    this._generationManager.manage(e, "iter_count");
                    this._presetManager.manage(e, "iter_count");
                });
            });

            e.createTab("Pipeline", PipelineEditor, (e) => {
                this._projectManager.manage(e, "pipeline");
            });

            e.createTab("Video Rendering", Form, (e) => {
                this._videoRenderer = e.createChild(VideoRendererEditor, (e) => {
                    this._presetManager.manage(e, "video_renderer");
                });

                this._videoParallelIndex = e.createField("Parallel index", NumberBox, (e) => {
                    e.minimum = 1;
                    e.step = 1;
                    e.value = 1;
                    this._presetManager.manage(e, "video_parallel_index");
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
                e.createField("Parallel index", NumberBox, (e) => {
                    e.minimum = 1;
                    e.step = 1;
                    e.value = 1;
                    this._presetManager.manage(e, "measuring_parallel_index");
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
            });

            e.createTab("Help", Column, (e) => {
                e.innerText = "Blah";
            });
        });

        // FIXME: Temporary
        this.onGenerationChange.connect((value) => console.log("GEN", value));
        this.onPresetChange.connect((value) => console.log("PST", value));
        this.onProjectChange.connect((value) => console.log("PRJ", value));
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
