import {Block} from "./scripts/base/block.js";
import {Checkbox} from "./scripts/base/checkbox.js";
import {Column} from "./scripts/base/column.js";
import {CANVAS_TOOLS, CanvasTool} from "./scripts/base/canvas_widget.js";
import {DockGroup} from "./scripts/base/dock_group.js";
import {Dropdown} from "./scripts/base/dropdown.js";
import {Form} from "./scripts/base/form.js";
import {ImageBox} from "./scripts/base/image_box.js";
import {MultiStateButton} from "./scripts/base/multi_state_button.js";
import {NumberBox} from "./scripts/base/number_box.js";
import {ProgressBar} from "./scripts/base/progress_bar.js";
import {Tabs} from "./scripts/base/tabs.js";
import {FieldManager} from "./scripts/core/field_manager.js";
import {Signal} from "./scripts/core/signal.js";
import {Timer} from "./scripts/core/timer.js";
import {Widget} from "./scripts/core/widget.js";
import {getRequest, postRequest} from "./scripts/utils/requests.js";
import {FSStoreBox} from "./scripts/fs_store_box.js";
import {ObjectForm} from "./scripts/object_form.js";
import {SettingsEditor} from "./scripts/settings_editor.js";
import {initializeData, pipelineModules, shared} from "./scripts/shared_data.js";

CANVAS_TOOLS.filter = class extends CanvasTool {
    static name = "Filter";
    static icon = "\u{f890}";
    static retained = true;

    makeUI(form) {
        this._filter = form.createField("Filter", Dropdown, (e) => {
            let names = {};
            let schemas = {};

            for (let [name, schema] of Object.entries(pipelineModules)) {
                if (name.startsWith("temporal.pipeline_modules.filtering")) {
                    names[name] = schema.name;
                    schemas[name] = schema;
                }
            }

            e.choices = names;
            e.style.width = "100%";
            e.onValueChange.connect((value) => {
                if (this._params) {
                    this._body.removeChild(this._params);
                }

                this._params = this._body.createChild(ObjectForm, (e) => {
                    for (let key of Object.keys(schemas[value].fields)) {
                        if (!["enabled", "preview"].includes(key)) {
                            e.manage(key);
                        }
                    }
                }, value);
            });
        });

        this._body = form.createChild(Block);

        // NOTE: To trigger the body creation
        this._filter.value = this._filter.value;
    }

    async onApply() {
        let image = new Image();
        image.addEventListener("load", () => {
            this._mainCtx.drawImage(image, 0, 0);
        });
        image.src = await postRequest("/temporal/module/execute", {
            "data": this._params.value,
            "image": this._mainCanvas.toDataURL("image/png"),
        });
    }
}

export class MainUI extends Widget {
    constructor() {
        super();

        this.onValueChange = new Signal();
        this.onProjectLoad = new Signal();
        this.onProjectChange = new Signal();
        this.onGenerationStart = new Signal();
        this.onGenerationStop = new Signal();
        this.onStateCheck = new Signal();

        this._manager = new FieldManager(this.onValueChange);

        this._stateTimer = new Timer(async () => {
            this.onStateCheck.fire(await getRequest("/temporal/execution/state"));
        }, 1.0);
        this.onGenerationStart.connect(() => this._stateTimer.start());
        this.onGenerationStop.connect(() => this._stateTimer.stop());

        this.style.height = "100%";
        this.style.position = "fixed";
        this.style.width = "100%";

        this.createChild(Column, (e) => {
            e.style.inset = "0";
            e.style.padding = "var(--layout-padding)";
            e.style.position = "absolute";

            e.createChild(MultiStateButton, (e) => {
                e.states = {stopped: "Generate", active: "Stop"};
                e.state = "stopped";
                e.style.minHeight = "calc(var(--widget-height) * 2)";
                e.onStateChange.connect(async (state) => {
                    if (state == "active") {
                        // FIXME: Kinda ugly, as it makes UI's responsiveness
                        // dependent on the server state, but we shouldn't start
                        // the state-changing timer until this call returns.
                        // ...
                        // (Yes, I hate all those "deferred" things that make
                        // my UX feel sluggish.)
                        await postRequest("/temporal/execution/generate", this._manager.value);

                        this.onGenerationStart.fire();
                    } else if (state == "stopped") {
                        this.onGenerationStop.fire();

                        await postRequest("/temporal/execution/interrupt");
                    }
                });
                this.onStateCheck.connect((state) => {
                    if (state.state == "stopping") {
                        e.enabled = false;
                    } else if (state.state == "stopped") {
                        e.state = "stopped";
                        e.enabled = true;
                    }
                });
            });

            e.createChild(ProgressBar, (e) => {
                e.visible = false;
                this.onGenerationStart.connect(() => {
                    e.value = 0;
                    e.total = 0;
                    e.text = "(Indeterminate)";
                    e.visible = true;
                });
                this.onGenerationStop.connect(() => {
                    e.visible = false;
                });
                this.onStateCheck.connect((state) => {
                    e.value = state.current_iteration;
                    e.total = state.total_iterations;
                    e.text = `${e.value} / ${e.total}`;
                });
            });

            this._image = e.createChild(ImageBox, (e) => {
                this.onStateCheck.connect((state) => {
                    if (state.preview) {
                        e.value = state.preview;
                    }
                });
            }, ["clear", "download", "fullscreen"]);

            e.createChild(MultiStateButton, (e) => {
                e.states = {normal: "Fullscreen", fullscreen: "Back to normal"};
                e.state = "normal";
                e.onStateChange.connect((state) => {
                    if (state == "fullscreen" && !document.fullscreenElement) {
                        document.documentElement.requestFullscreen();
                    } else if (state == "normal") {
                        document.exitFullscreen();
                    }
                });
            });
        });

        this.createChild(DockGroup, (e) => {
            e.createDock("\u{f53f}", "Project", Form, (e) => {
                e.createField("Preset", FSStoreBox, (e) => {
                    e.saveCallback = () => ({"data": this._manager.value});
                    e.onLoad.connect((value) => {
                        this._image.value = value.data.project.general.initial_image;
                    });
                }, "presets", ["refresh", "load", "save", "rename", "delete"]);

                e.createField("Project", FSStoreBox, (e) => {
                    e.onLoad.connect(async (value) => {
                        this._image.value = await getRequest(`/temporal/project/${e.value}/last_image`);

                        this.onProjectLoad.fire(value);
                    });
                }, "projects", ["refresh", "load", "rename", "delete"]);

                e.createField("Iteration count", NumberBox, (e) => {
                    e.minimum = 1;
                    e.step = 1;
                    e.value = 10;
                    this._manager.manage(e, "iter_count");
                });

                e.createRow((e) => {
                    e.createField("Load parameters", Checkbox, (e) => {
                        e.value = true;
                        this._manager.manage(e, "load_parameters");
                    });

                    e.createField("Continue from last iteration", Checkbox, (e) => {
                        e.value = true;
                        this._manager.manage(e, "continue_from_last_iteration");
                    });
                });

                // TODO
                this._hotReload = e.createField("Hot reload", Checkbox, (e) => {
                    e.value = false;
                });

                e.createChild(ObjectForm, (e) => {
                    e.manageAll();
                    e.onValueChange.connect(async (value) => {
                        shared.projectName = value.general.name;

                        this.onProjectChange.fire(value);
                    });
                    this.onProjectLoad.connect((project) => {
                        e.value = project;
                    });
                    this._manager.manage(e, "project");
                }, "temporal.project.Project");
            });

            e.createDock("\u{f013}", "System", Tabs, (e) => {
                e.createTab("Settings", SettingsEditor, (e) => {
                    e.onApply.connect(async (value) => {
                        await postRequest("/temporal/settings/apply", {
                            "data": value,
                        });
                    });
                });

                e.createTab("Help", Column, (e) => {
                    // TODO
                    e.innerText = "Blah";
                });
            });
        });
    }
}
customElements.define("main-ui", MainUI);

window.onload = async () => {
    await initializeData();

    document.body.appendChild(new MainUI());
};
