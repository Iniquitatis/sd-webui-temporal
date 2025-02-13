import {Button} from "../scripts/base/button.js";
import {Checkbox} from "../scripts/base/checkbox.js";
import {Column} from "../scripts/base/column.js";
import {Dropdown} from "../scripts/base/dropdown.js";
import {MultiStateToggle} from "../scripts/base/multi_state_toggle.js";
import {ReorderableAccordion} from "../scripts/base/reorderable_list.js";
import {Slider} from "../scripts/base/slider.js";
import {Tabs} from "../scripts/base/tabs.js";
import {FieldManager} from "../scripts/core/field_manager.js";
import {Signal} from "../scripts/core/signal.js";
import {createElement} from "../scripts/utils/dom.js";
import {ConfigurableParamEditor} from "../scripts/configurable_param_editor.js";
import {ImageMaskEditor} from "../scripts/image_mask_editor.js";
import {blendModes} from "../scripts/shared_data.js";

export class PipelineModuleEditor extends ReorderableAccordion {
    constructor(definition) {
        super();

        this.onValueChange = new Signal();
        this.onRemove = new Signal();

        this._manager = new FieldManager(this.onValueChange);
        this._manager.value = {id: definition.id, enabled: true};

        this._header.insertBefore(createElement(null, "input", (e) => {
            e.type = "checkbox";
            e.checked = true;
            e.addEventListener("change", () => {
                this._manager._value.enabled = e.checked;

                this.onValueChange.fire(this._value);
            });
            this._manager._onValueReceive.connect((value) => {
                e.checked = value.enabled;
            });
        }), this._header.firstChild.nextSibling);

        this._header.insertBefore(createElement(null, MultiStateToggle, (e) => {
            e.value = "on";
            this._manager.manage(e, "preview", (value) => value ? "on" : "off", (value) => value == "on");
        }, {"on": "\u{1f441}\u{fe0e}", "off": "\u{20e0}\u{fe0e}"}), this._header.lastChild);

        this.createChild(Column, (e) => {
            if (definition.is_filter) {
                e.createChild(Slider, (e) => {
                    e.label = "Amount"
                    e.minimum = 0.0;
                    e.maximum = 1.0;
                    e.step = 0.01;
                    e.value = 1.0;
                    this._manager.manage(e, "amount");

                    e.createChild(Checkbox, (e) => {
                        e.label = "Relative";
                        e.value = false;
                        e.style.width = "unset";
                        this._manager.manage(e, "amount_relative");
                    });
                });

                e.createChild(Dropdown, (e) => {
                    e.label = "Blend mode";
                    e.choices = blendModes;
                    this._manager.manage(e, "blend_mode", (value) => value.id, (value) => ({id: value}));
                });

                e.createChild(Tabs, (e) => {
                    e.createTab("Parameters", Column, (e) => {
                        for (let [id, param] of Object.entries(definition.parameters)) {
                            e.createChild(ConfigurableParamEditor, (e) => {
                                this._manager.manage(e, id);
                            }, param);
                        }
                    });

                    e.createTab("Mask", ImageMaskEditor, (e) => {
                        this._manager.manage(e, "mask");
                    });
                });
            } else {
                for (let [id, param] of Object.entries(definition.parameters)) {
                    e.createChild(ConfigurableParamEditor, (e) => {
                        this._manager.manage(e, id);
                    }, param);
                }
            }

            e.createChild(Button, (e) => {
                e.label = "\u{274c}\u{fe0e} Remove";
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
}
customElements.define("pipeline-module-editor", PipelineModuleEditor);
