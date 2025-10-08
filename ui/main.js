import {Block} from "/scripts/base/block.js";
import {Checkbox} from "/scripts/base/checkbox.js";
import {Column} from "/scripts/base/column.js";
import {CANVAS_TOOLS, CanvasTool} from "/scripts/base/canvas_widget.js";
import {DockGroup} from "/scripts/base/dock_group.js";
import {Dropdown} from "/scripts/base/dropdown.js";
import {Form} from "/scripts/base/form.js";
import {ImageBox} from "/scripts/base/image_box.js";
import {MultiStateButton} from "/scripts/base/multi_state_button.js";
import {NumberBox} from "/scripts/base/number_box.js";
import {ProgressBar} from "/scripts/base/progress_bar.js";
import {Tabs} from "/scripts/base/tabs.js";
import {FieldManager} from "/scripts/core/field_manager.js";
import {Signal} from "/scripts/core/signal.js";
import {Timer} from "/scripts/core/timer.js";
import {Widget} from "/scripts/core/widget.js";
import {defineElement} from "/scripts/utils/dom.js";
import {getRequest, postRequest} from "/scripts/utils/requests.js";
import {secondsToHHMMSS} from "/scripts/utils/time.js";
import {FSStoreBox} from "/scripts/fs_store_box.js";
import {ObjectForm} from "/scripts/object_form.js";
import {SettingsEditor} from "/scripts/settings_editor.js";
import {initializeData, pipelineModules} from "/scripts/shared_data.js";

CANVAS_TOOLS.filter = class extends CanvasTool {
    static name = "Filter";
    static icon = "\u{f890}";
    static retained = true;

    makeUI(form) {
        this._filter = form.createField("Filter", Dropdown, (e) => {
            let names = {};
            let schemas = {};

            for (let [name, schema] of Object.entries(pipelineModules)) {
                if (name.startsWith("modules.pipeline_modules.filtering")) {
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
                        if (!["enabled", "preview", "animation"].includes(key)) {
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
        image.src = await postRequest("/api/module/execute", {
            "data": this._params.value,
            "image": this._mainCanvas.toDataURL("image/png"),
        });
    }
}

export class MainUI extends Widget {
    static tag = "ce-main-ui";
    static css = `
        <self> {
            height: 100%;
            position: fixed;
            width: 100%;
        }

        <self> > ce-column {
            inset: 0;
            padding: var(--layout-padding);
            position: absolute;
        }

        <self> > ce-column > ce-multi-state-button:nth-of-type(1) {
            min-height: calc(var(--widget-height) * 2);
        }
    `;

    constructor() {
        super();

        this.onValueChange = new Signal();
        this.onProjectLoad = new Signal();
        this.onStateCheck = new Signal();

        this._manager = new FieldManager(this.onValueChange);

        this._stateTimer = new Timer(async () => {
            let state = await getRequest("/api/execution/state");

            if (state.state == "stopped") {
                this._stateTimer.stop();
            }

            this.onStateCheck.fire(state);
        }, 1.0);

        this.createChild(Column, (e) => {
            this._stateButton = e.createChild(MultiStateButton, (e) => {
                e.states = {stopped: "Generate", active: "Stop"};
                e.state = "stopped";
                e.onClick.connect(async () => {
                    switch (e.state) {
                        case "active": await this._start(); break;
                        case "stopped": await this._stop(); break;
                    }
                });
                this.onStateCheck.connect((state) => {
                    switch (state.state) {
                        case "active": e.state = "active"; break;
                        case "stopped": e.state = "stopped"; break;
                    }
                });
            });

            e.createChild(ProgressBar, (e) => {
                e.visible = false;
                this.onStateCheck.connect((state) => {
                    switch (state.state) {
                        case "active": {
                            e.value = state.current_iteration;
                            e.total = state.total_iterations;
                            e.text = `${e.value} / ${e.total} (${secondsToHHMMSS(state.eta)})`;
                            e.visible = true;
                        } break;

                        case "stopped": {
                            e.visible = false;
                        } break;
                    }
                });
            });

            e.createChild(ImageBox, (e) => {
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
                e.createField("Project", FSStoreBox, (e) => {
                    e.saveCallback = () => this._manager.value.project;
                    e.onLoad.connect((value) => {
                        this.onProjectLoad.fire(value);
                    });
                }, "projects", ["refresh", "load", "save", "rename", "delete"]);

                e.createField("Iteration count", NumberBox, (e) => {
                    e.minimum = 1;
                    e.step = 1;
                    e.value = 10;
                    this._attachToHotReload(e);
                    this._manager.manage(e, "iterations");
                });

                this._hotReload = e.createField("Hot reload", Checkbox, (e) => {
                    e.value = false;
                });

                e.createChild(ObjectForm, (e) => {
                    e.manageAll();
                    this.onProjectLoad.connect((project) => {
                        e.value = project;
                    });
                    this._attachToHotReload(e);
                    this._manager.manage(e, "project");
                }, "modules.project.Project");
            });

            e.createDock("\u{f013}", "System", Tabs, (e) => {
                e.createTab("Settings", SettingsEditor, (e) => {
                    e.onApply.connect(async (value) => {
                        await postRequest("/api/settings/apply", {
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

    _attachToHotReload(widget) {
        widget.onValueChange.connect(async () => {
            // NOTE: Might be called multiple times in parallel because of async
            // nature (for example, when user spams some toggle), but there's
            // no clean way to cancel the current "await". Every other solution
            // (queueing, early return based on a state flag, etc.) would lead
            // to worse UX.
            if (!this._hotReload.value) return;

            await this._stop();
            await this._start();
        });
    }

    async _start() {
        this._stateButton.enabled = false;

        await postRequest("/api/execution/generate", this._manager.value);

        this._stateTimer.start();

        this._stateButton.enabled = true;
    }

    async _stop() {
        this._stateButton.enabled = false;

        await postRequest("/api/execution/interrupt");

        this._stateTimer.stop();

        this._stateButton.enabled = true;
    }
}
defineElement(MainUI);

window.onload = async () => {
    await initializeData();

    document.body.appendChild(new MainUI());
};
