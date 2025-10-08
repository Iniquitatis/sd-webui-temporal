import {Button} from "/scripts/base/button.js";
import {Checkbox} from "/scripts/base/checkbox.js";
import {Column} from "/scripts/base/column.js";
import {ImageBox} from "/scripts/base/image_box.js";
import {ChoiceListEditor} from "/scripts/base/list_editor.js";
import {MultiStateToggle} from "/scripts/base/multi_state_toggle.js";
import {ReorderableAccordion} from "/scripts/base/reorderable_list.js";
import {Row} from "/scripts/base/row.js";
import {Tabs} from "/scripts/base/tabs.js";
import {VideoBox} from "/scripts/base/video_box.js";
import {FieldManager} from "/scripts/core/field_manager.js";
import {Signal} from "/scripts/core/signal.js";
import {defineElement} from "/scripts/utils/dom.js";
import {deepCopy, mapValues} from "/scripts/utils/object.js";
import {postRequest} from "/scripts/utils/requests.js";
import {boolToString, stringToBool} from "/scripts/utils/types.js";
import {AnimationEditor} from "/scripts/animation_editor.js";
import {ObjectForm} from "/scripts/object_form.js";
import {pipelineModules, pipelineModuleIcons} from "/scripts/shared_data.js";

class PipelineModuleEditor extends ReorderableAccordion {
    static tag = "ce-pipeline-module-editor";
    static css = `
        <self> .sample {
            height: 12rem;
        }
    `;

    constructor(type) {
        super();

        this.onValueChange = new Signal();
        this.onDuplicateRequest = new Signal();
        this.onRemoveRequest = new Signal();

        this._manager = new FieldManager(this.onValueChange);
        this._manager.value.__type__ = type;

        let schema = pipelineModules[type];

        this.label = `${pipelineModuleIcons[type]} ${schema.name}`;

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
                this._sampleBox = e.createChild(ImageBox, (e) => {
                    e.classList.add("sample");
                    this.onValueChange.connect(async () => {
                        await this._updateSample();
                    });
                });
            }

            // FIXME: Should be handled differently. Probably fields would have
            // to contain "display" property in order for them to be shown at
            // all.
            let filteredKeys = Object.keys(schema.fields).filter((key) => {
                return !["enabled", "preview", "animation", "amount", "blend_mode", "mask"].includes(key);
            });

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

                    // TODO: Filter out animatable fields
                    // FIXME: Shown based on a wrong assumption that only
                    // filters or visualizables are animatable
                    e.createTab("Animation", AnimationEditor, (e) => {
                        this._manager.manage(e, "animation");
                    }, schema);

                    // FIXME: Needs rework--probably send these into the main
                    // preview area instead. Then there won't be any need in IDs
                    // here, as these things won't store any state anymore and
                    // will be just inputs instead.
                    if (schema.is_visualizable) {
                        e.createTab("Viz", Column, (e) => {
                            e.createChild(Button, (e) => {
                                e.label = "Render";
                                e.onClick.connect(() => {
                                    e.enabled = false;

                                    // this._visualization.value = await postRequest(`/api/module/${shared.projectName}/${this._manager.value.__id__}/visualize`);

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
                    e.onClick.connect(() => {
                        this.onDuplicateRequest.fire(deepCopy(this.value));
                    });
                });

                e.createChild(Button, (e) => {
                    e.label = "\u{f2ed} Remove";
                    e.onClick.connect(() => {
                        this.onRemoveRequest.fire();
                    });
                });
            });
        });

        this._updateSample();
    }

    get value() {
        return this._manager.value;
    }

    set value(value) {
        this._manager.value = value;
    }

    async _updateSample() {
        if (!this._sampleBox) return;

        this._sampleBox.value = await postRequest("/api/module/sample", {
            "data": this._manager.value,
            "size": [256, 256],
        });
    }
}
defineElement(PipelineModuleEditor);

export class PipelineModuleList extends ChoiceListEditor {
    static tag = "ce-pipeline-module-list";

    constructor() {
        super(PipelineModuleEditor, mapValues(pipelineModules, (_, schema) => {
            return `${pipelineModuleIcons[schema.type]} ${schema.name}`;
        }));
    }

    getArgsFromItem(item) {
        return [item.__type__];
    }

    getArgsFromChoice(choice) {
        return [choice];
    }
}
defineElement(PipelineModuleList);
