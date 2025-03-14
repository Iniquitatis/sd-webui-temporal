import {Accordion} from "../scripts/base/accordion.js";
import {Button} from "../scripts/base/button.js";
import {Checkbox} from "../scripts/base/checkbox.js";
import {Column} from "../scripts/base/column.js";
import {Dropdown} from "../scripts/base/dropdown.js";
import {Form} from "../scripts/base/form.js";
import {ImageBox} from "../scripts/base/image_box.js";
import {MultiStateToggle} from "../scripts/base/multi_state_toggle.js";
import {ReorderableAccordion} from "../scripts/base/reorderable_list.js";
import {Slider} from "../scripts/base/slider.js";
import {Tabs} from "../scripts/base/tabs.js";
import {VideoBox} from "../scripts/base/video_box.js";
import {FieldManager} from "../scripts/core/field_manager.js";
import {Signal} from "../scripts/core/signal.js";
import {createElement} from "../scripts/utils/dom.js";
import {getRequest, postRequest} from "../scripts/utils/requests.js";
import {ImageMaskEditor} from "../scripts/image_mask_editor.js";
import {ParamForm} from "../scripts/param_form.js";
import {blendModes} from "../scripts/shared_data.js";

export class PipelineModuleEditor extends ReorderableAccordion {
    constructor(definition) {
        super();

        this.onValueChange = new Signal();
        this.onRemove = new Signal();

        this._manager = new FieldManager(this.onValueChange);
        this._manager.value = {__type__: definition.type, enabled: true};
        this._manager.onFieldChange.connect(async (key, value) => {
            await this._ensureID();

            await postRequest(`/temporal/object/${this._manager.value.__id__}/field`, {
                "key": key,
                "value": value,
            });
        });

        this._header.insertBefore(createElement(null, Checkbox, (e) => {
            e.value = true;
            this._manager.manage(e, "enabled");
        }), this._header.firstChild.nextSibling);

        this._header.insertBefore(createElement(null, MultiStateToggle, (e) => {
            e.states = {on: "\u{f06e}", off: "\u{f070}"};
            e.value = "on";
            this._manager.manage(e, "preview", (value) => value ? "on" : "off", (value) => value == "on");
        }), this._header.lastChild);

        this.createChild(Column, (e) => {
            if (definition.is_sampleable) {
                e.createChild(ImageBox, (e) => {
                    e.style.height = "12rem";
                    this.onValueChange.connect(async (value) => {
                        await this._ensureID();

                        e.value = await postRequest(`/temporal/module/${this._manager.value.__id__}/render_sample`);
                    });
                });
            }

            e.createChild(Tabs, (e) => {
                if (Object.keys(definition.parameters).length > 0) {
                    e.createTab("Parameters", Column, (e) => {
                        e.createChild(ParamForm, null, definition.parameters, this._manager);

                        if (definition.type.startsWith("temporal.pipeline_modules.measuring")) {
                            e.createChild(Button, (e) => {
                                e.label = "Plot";
                                e.onClick.connect(async () => {
                                    e.enabled = false;

                                    await this._ensureID();

                                    this._graph.value = await postRequest(`/temporal/module/${this._manager.value.__id__}/render_graph`);

                                    e.enabled = true;
                                });
                            });

                            this._graph = e.createChild(ImageBox, null, ["fullscreen"]);
                        }

                        if (definition.type.startsWith("temporal.pipeline_modules.tool.video_rendering")) {
                            e.createChild(Button, (e) => {
                                e.label = "Render";
                                e.onClick.connect(async () => {
                                    e.enabled = false;

                                    await this._ensureID();

                                    this._video.value = await postRequest(`/temporal/module/${this._manager.value.__id__}/render_video`);

                                    e.enabled = true;
                                });
                            });

                            this._video = e.createChild(VideoBox, null, ["download", "fullscreen"]);
                        }
                    });
                }

                if (definition.is_filter) {
                    e.createTab("Blending", Form, (e) => {
                        e.createField("Amount", Slider, (e) => {
                            e.minimum = 0.0;
                            e.maximum = 1.0;
                            e.step = 0.01;
                            e.value = 1.0;
                            this._manager.manage(e, "amount");
                        });

                        e.createField("Blend mode", Dropdown, (e) => {
                            e.choices = blendModes;
                            this._manager.manage(e, "blend_mode", (value) => value.__type__, (value) => ({__type__: value}));
                        });

                        e.createChild(Accordion, (e) => {
                            e.label = "Mask";

                            e.createChild(ImageMaskEditor, (e) => {
                                this._manager.manage(e, "mask");
                            });
                        });
                    });
                }
            });

            e.createChild(Button, (e) => {
                e.label = "\u{f2ed} Remove";
                e.onClick.connect(() => {
                    this.parentElement.removeChild(this);

                    this.onRemove.fire();
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
