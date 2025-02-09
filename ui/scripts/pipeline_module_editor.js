import {Button} from "../scripts/base/button.js";
import {Checkbox} from "../scripts/base/checkbox.js";
import {Column} from "../scripts/base/column.js";
import {Dropdown} from "../scripts/base/dropdown.js";
import {MultiStateToggle} from "../scripts/base/multi_state_toggle.js";
import {ReorderableAccordion} from "../scripts/base/reorderable_list.js";
import {Slider} from "../scripts/base/slider.js";
import {Signal} from "../scripts/core/signal.js";
import {createElement} from "../scripts/utils/dom.js";
import {ConfigurableParamEditor} from "../scripts/configurable_param_editor.js";
import {blendModes} from "../scripts/test_data.js";

export class PipelineModuleEditor extends ReorderableAccordion {
    constructor(id, definition) {
        super(id);

        this.module = {id: id, enabled: true};

        this.label = `${definition.icon} ${definition.name}`;

        this._header.insertBefore(createElement(null, "input", (e) => {
            e.type = "checkbox";
            e.checked = true;
            e.addEventListener("change", () => {
                this.module.enabled = e.checked;

                this.onValueChange.fire(this.module);
            });
        }), this._header.firstChild.nextSibling);

        this._header.insertBefore(createElement(null, MultiStateToggle, (e) => {
            e.value = "on";
            e.onValueChange.connect((value) => {
                this.module.preview = value == "on";

                this.onValueChange.fire(this.module);
            });
        }, {"on": "\u{1f441}", "off": "\u{25ce}"}), this._header.lastChild);

        this.createChild(Column, (e) => {
            if (definition.is_filter) {
                e.createChild(Slider, (e) => {
                    e.label = "Amount"
                    e.minimum = 0.0;
                    e.maximum = 1.0;
                    e.step = 0.01;
                    e.value = 1.0;
                    e.onValueChange.connect((value) => {
                        this.module.amount = value;

                        this.onValueChange.fire(this.module);
                    });

                    e.createChild(Checkbox, (e) => {
                        e.label = "Relative";
                        e.value = false;
                        e.style.width = "unset";
                        e.onValueChange.connect((value) => {
                            this.module.relative = value;

                            this.onValueChange.fire(this.module);
                        });
                    });
                });

                e.createChild(Dropdown, (e) => {
                    e.label = "Blend mode";
                    e.choices = blendModes;
                    e.onValueChange.connect((value) => {
                        this.module.blend_mode = value;

                        this.onValueChange.fire(this.module);
                    });
                });
            }

            for (let [id, param] of Object.entries(definition.parameters)) {
                e.createChild(ConfigurableParamEditor, (e) => {
                    e.onValueChange.connect((value) => {
                        this.module[id] = value;

                        this.onValueChange.fire(this.module);
                    });
                }, param);
            }

            e.createChild(Button, (e) => {
                e.label = "\u{274c} Remove";
                e.onClick.connect(() => {
                    this.parentElement.removeChild(this)

                    this.onRemove.fire();
                });
            });
        });

        this.onValueChange = new Signal();
        this.onRemove = new Signal();
    }
}
customElements.define("pipeline-module-editor", PipelineModuleEditor);
