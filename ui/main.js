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
            e.style.height = "calc(var(--widget-height) * 2)";
            e.onStateChange.connect((state) => {
                if (state == "active") {
                    this._progressBar.value = 0;
                    this._progressBar.total = 0;
                    this._progressBar.text = "(Indeterminate)";
                    this._progressBar.style.display = null;

                    this._progressInterval = window.setInterval(async () => {
                        await getRequest("/temporal/state", (result) => {
                            if (result.state == "active") {
                                this._progressBar.value = result.current_iteration;
                                this._progressBar.total = result.total_iterations;
                                this._progressBar.text = `${this._progressBar.value} / ${this._progressBar.total}`;
                            } else if (result.state == "stopped") {
                                e.state = "stopped";
                            }
                        });

                        await getRequest("/temporal/preview", (result) => {
                            if (!result) return;

                            this._preview.value = `data:image/png;base64,${result}`;
                        });
                    }, 1000);

                    postRequest("/temporal/generate", generation);
                } else if (state == "stopped") {
                    this._progressBar.style.display = "none";

                    window.clearInterval(this._progressInterval);
                    this._progressInterval = null;

                    postRequest("/temporal/interrupt");
                }
            });
        }, {"stopped": "Generate", "active": "Stop"});

        this._progressBar = this.createChild(ProgressBar, (e) => {
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

        this.createChild(FSStoreBox, (e) => {
            e.label = "Project";
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

        this.createChild(Tabs, (e) => {
            e.createTab("General", Column, (e) => {
                this._name = e.createChild(TextBox, (e) => {
                    e.label = "Name";
                    e.onValueChange.connect((value) => {
                        generation.name = value;
                        preset.name = value;
                    });
                });

                e.createChild(TextArea, (e) => {
                    e.label = "Description";
                });

                this._loadParameters = e.createChild(Checkbox, (e) => {
                    e.label = "Load parameters";
                    e.value = true;
                    e.onValueChange.connect((value) => {
                        generation.load_parameters = value;
                        preset.load_parameters = value;
                    });
                });

                this._continueFromLastFrame = e.createChild(Checkbox, (e) => {
                    e.label = "Continue from last frame";
                    e.value = true;
                    e.onValueChange.connect((value) => {
                        generation.continue_from_last_frame = value;
                        preset.continue_from_last_frame = value;
                    });
                });

                this._iterCount = e.createChild(NumberBox, (e) => {
                    e.label = "Iteration count";
                    e.minimum = 1;
                    e.step = 1;
                    e.value = 10;
                    e.onValueChange.connect((value) => {
                        generation.iter_count = value;
                        preset.iter_count = value;
                    });
                });
            });

            e.createTab("Processing", Column, (e) => {
                this._processing = e.createChild(ProcessingParamsEditor, (e) => {
                    e.onValueChange.connect((value) => {
                        generation.project.general.parameters = value;
                        preset.project.general.parameters = value;
                    });
                    // FIXME: Temporary
                    e.value = {
                        "clip_skip": 1,
                        "positive_prompts": ["female, portrait, forest, cinematic, backlighting"],
                        "negative_prompts": ["male, gray theme, 2d"],
                        "width": 768,
                        "height": 1152,
                        "sampler": "DPM++ 3M SDE",
                        "scheduler": "DDIM",
                        "steps": 10,
                        "cfg": 2.5,
                        "strength": 1.0,
                        "seeds": [-1],
                    };
                });
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
                        e.value = {
                            "factor": 0.16,
                            "noise": {
                                "mode": "ridge",
                                "scale": 69,
                                "detail": 4.51,
                                "lacunarity": 2.87,
                                "persistence": 0.42,
                                "seed": 31337,
                                "use_global_seed": true,
                            },
                        };
                        // FIXME: Temporary
                        e.onValueChange.connect((value) => console.log(value));
                    });
                });

                this._parallel = e.createChild(NumberBox, (e) => {
                    e.label = "Parallel";
                    e.minimum = 1;
                    e.step = 1;
                    e.value = 1;
                    e.onValueChange.connect((value) => {
                        generation.project.general.parallel = value;
                        preset.project.general.parallel = value;
                    });
                });

                this._pipelineModules = e.createChild(ModuleList, (e) => {
                    e.onValueChange.connect((value) => {
                        generation.project.modules = value;
                        preset.project.modules = value;
                    });
                    // FIXME: Temporary
                    e.value = [
                        {
                            "id": "temporal.pipeline_modules.painting.color.ColorPaintingModule",
                            "enabled": true,
                            "preview": false,
                            "amount": 0.3,
                            "blend_mode": {"id": "temporal.blend_modes.MultiplyBlendMode"},
                            "color": {r: 0.25, g: 0.5, b: 0.9, a: 1.0},
                        },
                        {
                            "id": "temporal.pipeline_modules.neural.processing.ProcessingModule",
                            "enabled": true,
                            "preview": false,
                        },
                        {
                            "id": "temporal.pipeline_modules.tool.saving.SavingModule",
                            "enabled": true,
                            "preview": true,
                            "archive_mode": true,
                        },
                    ];
                    // FIXME: Temporary
                    e.onValueChange.connect((value) => console.log(value, generation));
                }, "Add module", PipelineModuleEditor, mapObject(pipelineModules, (_, module) => `${module.icon} ${module.name}`), pipelineModules);

                this._animation = e.createChild(CodeArea, (e) => {
                    e.label = "Animation",
                    e.onValueChange.connect((value) => {
                        generation.project.animation = {code: value};
                        preset.project.animation = {code: value};
                    });
                });
            });

            e.createTab("Video Rendering", Column, (e) => {
                this._videoRenderer = e.createChild(VideoRendererEditor, (e) => {
                    e.onValueChange.connect((value) => {
                        preset.video_renderer = value;
                    });
                });
            });

            e.createTab("Measuring", Column, (e) => {
                this._measuringParallelIndex = e.createChild(NumberBox, (e) => {
                    e.label = "Parallel index";
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

window.onload = async () => {
    await getRequest("/temporal/blend_modes", (result) => {
        for (let [k, v] of Object.entries(result)) {
            blendModes[k] = v;
        }
    });

    await getRequest("/temporal/models", (result) => {
        for (let model of result) {
            models[model] = model;
        }
    });

    await getRequest("/temporal/pipeline_modules", (result) => {
        for (let [k, v] of Object.entries(result)) {
            pipelineModules[k] = v;
        }
    });

    await getRequest("/temporal/presets", (result) => {
        for (let preset of result) {
            presets[preset] = preset;
        }
    });

    await getRequest("/temporal/projects", (result) => {
        for (let project of result) {
            projects[project] = project;
        }
    });

    await getRequest("/temporal/samplers", (result) => {
        for (let sampler of result) {
            samplers[sampler] = sampler;
        }
    });

    await getRequest("/temporal/schedulers", (result) => {
        for (let scheduler of result) {
            schedulers[scheduler] = scheduler;
        }
    });

    await getRequest("/temporal/vaes", (result) => {
        for (let vae of result) {
            vaes[vae] = vae;
        }
    });

    await getRequest("/temporal/video_filters", (result) => {
        for (let [k, v] of Object.entries(result)) {
            videoFilters[k] = v;
        }
    });

    document.body.appendChild(new MainUI());
};
