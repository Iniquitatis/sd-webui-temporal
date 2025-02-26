import {Accordion} from "../scripts/base/accordion.js";
import {Button} from "../scripts/base/button.js";
import {Checkbox} from "../scripts/base/checkbox.js";
import {Dropdown} from "../scripts/base/dropdown.js";
import {Form} from "../scripts/base/form.js";
import {MultiStateToggle} from "../scripts/base/multi_state_toggle.js";
import {ReorderableAccordion} from "../scripts/base/reorderable_list.js";
import {Slider} from "../scripts/base/slider.js";
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
        this._manager.value = {__type__: definition.type, enabled: true};

        this._header.insertBefore(createElement(null, Checkbox, (e) => {
            e.value = true;
            this._manager.manage(e, "enabled");
        }), this._header.firstChild.nextSibling);

        this._header.insertBefore(createElement(null, MultiStateToggle, (e) => {
            e.states = {on: "\u{f06e}", off: "\u{f070}"};
            e.value = "on";
            this._manager.manage(e, "preview", (value) => value ? "on" : "off", (value) => value == "on");
        }), this._header.lastChild);

        this.createChild(Form, (e) => {
            if (definition.is_filter) {
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
            }

            for (let [id, param] of Object.entries(definition.parameters)) {
                e.createField(param.name, ConfigurableParamEditor, (e) => {
                    this._manager.manage(e, id);
                }, param);
            }

            if (definition.is_filter) {
                e.createChild(Accordion, (e) => {
                    e.label = "Mask";

                    e.createChild(ImageMaskEditor, (e) => {
                        this._manager.manage(e, "mask");
                    });
                });
            }

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
}
customElements.define("pipeline-module-editor", PipelineModuleEditor);
