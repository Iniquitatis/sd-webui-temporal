import {Button} from "../scripts/base/button.js";
import {Checkbox} from "../scripts/base/checkbox.js";
import {Column} from "../scripts/base/column.js";
import {ImageBox} from "../scripts/base/image_box.js";
import {MultiStateToggle} from "../scripts/base/multi_state_toggle.js";
import {ReorderableAccordion} from "../scripts/base/reorderable_list.js";
import {Row} from "../scripts/base/row.js";
import {Tabs} from "../scripts/base/tabs.js";
import {VideoBox} from "../scripts/base/video_box.js";
import {FieldManager} from "../scripts/core/field_manager.js";
import {Signal} from "../scripts/core/signal.js";
import {deepCopy} from "../scripts/utils/object.js";
import {getRequest, postRequest} from "../scripts/utils/requests.js";
import {boolToString, stringToBool} from "../scripts/utils/types.js";
import {ObjectForm} from "../scripts/object_form.js";
import {objectTypes} from "../scripts/shared_data.js";

export class PipelineModuleEditor extends ReorderableAccordion {
    constructor(type) {
        super();

        this.onValueChange = new Signal();
        this.onRemove = new Signal();

        this._manager = new FieldManager(this.onValueChange);
        this._manager.value.__type__ = type;

        (async () => {
            this.enabled = false;

            await this._ensureID()

            this.enabled = true;
        })();

        let schema = objectTypes[type];

        this.createBeforeLabel(Checkbox, (e) => {
            e.value = schema.fields.enabled.default;
            this._manager.manage(e, "enabled");
        });

        this.createAfterLabel(MultiStateToggle, (e) => {
            e.states = {true: "\u{f06e}", false: "\u{f070}"};
            e.value = boolToString(schema.fields.preview.default);
            this._manager.manage(e, "preview", boolToString, stringToBool);
        });

        this.createChild(Column, (e) => {
            if (schema.is_sampleable) {
                e.createChild(ImageBox, (e) => {
                    e.style.height = "12rem";
                    this.onValueChange.connect(async (value) => {
                        await this._ensureID();

                        e.value = await postRequest("/temporal/module/sample", {
                            "data": this._manager.value,
                            "size": [256, 256],
                        });
                    });
                });
            }

            let filteredKeys = [];

            for (let key of Object.keys(schema.fields)) {
                if (!["enabled", "preview", "amount", "blend_mode", "mask"].includes(key)) {
                    filteredKeys.push(key);
                }
            }

            if (schema.is_filter || schema.is_visualizable) {
                e.createChild(Tabs, (e) => {
                    if (filteredKeys.length > 0) {
                        e.createTab("Parameters", ObjectForm, (e) => {
                            e.manageMultiple(filteredKeys);
                        }, schema.type, this._manager);
                    }

                    if (schema.is_filter) {
                        e.createTab("Blending", ObjectForm, (e) => {
                            e.manageMultiple(["amount", "blend_mode", "mask"]);
                        }, schema.type, this._manager);
                    }

                    if (schema.is_visualizable) {
                        e.createTab("Viz", Column, (e) => {
                            e.createChild(Button, (e) => {
                                e.label = "Render";
                                e.onClick.connect(async () => {
                                    await this._ensureID();

                                    e.enabled = false;

                                    this._visualization.value = await postRequest(`/temporal/module/${this._manager.value.__id__}/visualize`);

                                    e.enabled = true;
                                });
                            });

                            this._visualization = e.createChild(schema.visualization_type == "image" ? ImageBox : VideoBox, null, ["download", "fullscreen"]);
                        });
                    }
                });
            } else if (filteredKeys.length > 0) {
                e.createChild(ObjectForm, (e) => {
                    e.manageMultiple(filteredKeys);
                }, schema.type, this._manager);
            } else {
                e.createChild("span", (e) => {
                    e.innerText = "This module has no configurable parameters.";
                });
            }

            e.createChild(Row, (e) => {
                e.createChild(Button, (e) => {
                    e.label = "\u{f0c5} Duplicate";
                    e.style.width = "100%";
                    e.onClick.connect(() => {
                        let newValue = deepCopy(this.value);
                        delete newValue.__id__;
                        // FIXME: Hacky. Should somehow interact with the
                        // ModuleList (which is one level higher), not
                        // ReorderableList.
                        this.parentElement.parentElement._createModule(type, newValue);
                    });
                });

                e.createChild(Button, (e) => {
                    e.label = "\u{f2ed} Remove";
                    e.style.width = "100%";
                    e.onClick.connect(() => {
                        this.parentElement.removeChild(this);

                        this.onRemove.fire();
                    });
                });
            });
        });
    }

    get value() {
        return this._manager.value;
    }

    set value(value) {
        this._manager.value = value;
    }

    async _ensureID() {
        if (this._manager.value.__id__) return;
        this._manager.value.__id__ = await getRequest("/temporal/utils/uuid");
    }
}
customElements.define("pipeline-module-editor", PipelineModuleEditor);
