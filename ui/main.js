import {Button} from "./scripts/base/button.js";
import {CanvasBox} from "./scripts/base/canvas_box.js";
import {Column} from "./scripts/base/column.js";
import {DockGroup} from "./scripts/base/dock_group.js";
import {Form} from "./scripts/base/form.js";
import {MultiStateButton} from "./scripts/base/multi_state_button.js";
import {NumberBox} from "./scripts/base/number_box.js";
import {ProgressBar} from "./scripts/base/progress_bar.js";
import {Row} from "./scripts/base/row.js";
import {Tabs} from "./scripts/base/tabs.js";
import {VideoBox} from "./scripts/base/video_box.js";
import {FieldManager} from "./scripts/core/field_manager.js";
import {Signal} from "./scripts/core/signal.js";
import {Timer} from "./scripts/core/timer.js";
import {Widget} from "./scripts/core/widget.js";
import {getRequest, postRequest} from "./scripts/utils/requests.js";
import {FSStoreBox} from "./scripts/fs_store_box.js";
import {OptionsEditor} from "./scripts/options_editor.js";
import {ProjectEditor} from "./scripts/project_editor.js";
import {SessionEditor} from "./scripts/session_editor.js";
import {initializeData, presets, projects} from "./scripts/shared_data.js";
import {VideoRendererEditor} from "./scripts/video_renderer_editor.js";

export class MainUI extends Widget {
    constructor() {
        super();

        this.onGenerationChange = new Signal();
        this.onPresetChange = new Signal();
        this.onGenerationStart = new Signal();
        this.onGenerationStop = new Signal();
        this.onStateCheck = new Signal();
        this.onNewPreview = new Signal();
        this.onVideoRenderStart = new Signal();
        this.onVideoRender = new Signal();

        this._generationManager = new FieldManager(this.onGenerationChange);
        this._presetManager = new FieldManager(this.onPresetChange);

        this._stateTimer = new Timer(async () => {
            this.onStateCheck.fire(await getRequest("/temporal/state"));

            let preview = await getRequest("/temporal/preview");

            if (preview) {
                this.onNewPreview.fire(preview);
            }
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
                        this.onGenerationStart.fire();

                        await postRequest("/temporal/generate", this._generationManager.value);
                    } else if (state == "stopped") {
                        this.onGenerationStop.fire();

                        await postRequest("/temporal/interrupt");
                    }
                });
                this.onStateCheck.connect((state) => {
                    if (state.state == "stopped") {
                        e.state = "stopped";
                    }
                });
            });

            e.createChild(ProgressBar, (e) => {
                e.style.display = "none";
                this.onGenerationStart.connect(() => {
                    e.value = 0;
                    e.total = 0;
                    e.text = "(Indeterminate)";
                    e.style.display = null;
                });
                this.onGenerationStop.connect(() => {
                    e.style.display = "none";
                });
                this.onStateCheck.connect((state) => {
                    e.value = state.current_iteration;
                    e.total = state.total_iterations;
                    e.text = `${e.value} / ${e.total}`;
                });
            });

            this._image = e.createChild(CanvasBox, (e) => {
                e._element.width = 512;
                e._element.height = 512;
                this._generationManager.manage(e, "image");
                this._presetManager.manage(e, "image");
                this.onNewPreview.connect((preview) => {
                    e.value = preview;
                });
            });

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
                    e.entries = presets;
                    e.saveCallback = () => this._presetManager.value;
                    e.onLoad.connect((value) => {
                        let newValue = this._generationManager.value;
                        newValue.image = value.project.general.initial_image;
                        newValue.project = value;

                        this._generationManager.value = newValue;
                        this._presetManager.value = value;
                    });
                }, "presets", ["refresh", "load", "save", "rename", "delete"]);

                e.createField("Project", FSStoreBox, (e) => {
                    e.entries = projects;
                    e.onLoad.connect(async (value) => {
                        let metadata = await postRequest("/temporal/project_metadata", {
                            "name": value.general.name,
                            "include_last_image": true,
                        });

                        let newValue = this._generationManager.value;
                        newValue.image = metadata.last_image;
                        newValue.project = value;

                        this._generationManager.value = newValue;
                    });
                }, "projects", ["refresh", "load", "rename", "delete"]);

                e.createChild(SessionEditor, (e) => {
                    this._generationManager.manage(e, "session");
                    this._presetManager.manage(e, "session");
                });

                e.createChild(ProjectEditor, (e) => {
                    e.onImageSizeChange.connect((value) => {
                        // FIXME: Accesses private stuff
                        this._image._element.width = value.x;
                        this._image._element.height = value.y;
                    });
                    this._generationManager.manage(e, "project");
                    this._presetManager.manage(e, "project");
                });

                // FIXME: Everything below has to be reworked, but it depends on
                // the preset system currently
                e.createChild("hr");

                e.createChild(Tabs, (e) => {
                    e.createTab("Video Rendering", Form, (e) => {
                        this._videoRenderer = e.createChild(VideoRendererEditor, (e) => {
                            this._presetManager.manage(e, "video_renderer");
                        });

                        e.createChild(Row, (e) => {
                            this.onVideoRenderStart.connect(() => {
                                for (let button of e.childNodes) {
                                    button.classList.add("disabled");
                                }
                            });
                            this.onVideoRender.connect((data) => {
                                for (let button of e.childNodes) {
                                    button.classList.remove("disabled");
                                }
                            });

                            for (let type of ["draft", "final"]) {
                                e.createChild(Button, (e) => {
                                    e.label = `Render ${type}`;
                                    e.style.width = "100%";
                                    e.onClick.connect(async () => {
                                        this.onVideoRenderStart.fire();
                                        this.onVideoRender.fire(await postRequest("/temporal/render_video", {
                                            "type": type,
                                            "data": this._videoRenderer.value,
                                        }));
                                    });
                                });
                            }
                        });

                        e.createChild(VideoBox, (e) => {
                            this.onVideoRender.connect((data) => {
                                e.value = data;
                            });
                        });
                    });

                    e.createTab("Measuring", Form, (e) => {
                        e.createChild(Button, (e) => {
                            e.label = "Render graphs";
                        });
                    });
                });
            });

            e.createDock("\u{f013}", "System", Tabs, (e) => {
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
        });

        // FIXME: Temporary
        this.onGenerationChange.connect((value) => console.log("GEN", value));
        this.onPresetChange.connect((value) => console.log("PST", value));
    }
}
customElements.define("main-ui", MainUI);

window.onload = async () => {
    await initializeData();

    document.body.appendChild(new MainUI());
};
