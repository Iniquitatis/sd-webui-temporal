import {Checkbox} from "../scripts/base/checkbox.js";
import {CodeArea} from "../scripts/base/code_area.js";
import {ColorPicker} from "../scripts/base/color_picker.js";
import {Dropdown} from "../scripts/base/dropdown.js";
import {ImageBox} from "../scripts/base/image_box.js";
import {NumberBox} from "../scripts/base/number_box.js";
import {Radio} from "../scripts/base/radio.js";
import {Slider} from "../scripts/base/slider.js";
import {TextArea} from "../scripts/base/text_area.js";
import {TextBox} from "../scripts/base/text_box.js";
import {Signal} from "../scripts/core/signal.js";
import {Widget} from "../scripts/core/widget.js";
import {SeedBox} from "../scripts/seed_box.js";

export class ConfigurableParamEditor extends Widget {
    constructor(definition) {
        super();

        let editor = null;

        switch (definition.type) {
            case "bool": {
                editor = this.createChild(Checkbox, (e) => {
                    e.label = definition.name;
                    e.value = definition.value;
                });
            } break;

            case "int": {
                editor = this.createChild(definition.ui_type == "slider" ? Slider : NumberBox, (e) => {
                    e.label = definition.name;
                    e.minimum = definition.minimum ?? undefined;
                    e.maximum = definition.maximum ?? undefined;
                    e.step = definition.step ?? 1;
                    e.value = definition.default ?? e.minimum;
                });
            } break;

            case "float": {
                editor = this.createChild(definition.ui_type == "slider" ? Slider : NumberBox, (e) => {
                    e.label = definition.name;
                    e.minimum = definition.minimum ?? undefined;
                    e.maximum = definition.maximum ?? undefined;
                    e.step = definition.step ?? 0.1;
                    e.value = definition.default ?? e.minimum;
                });
            } break;

            case "string": {
                editor = this.createChild(definition.ui_type == "code" ? CodeArea : definition.ui_type == "area" ? TextArea : TextBox, (e) => {
                    e.label = definition.name;
                    e.value = definition.default ?? "";
                });
            } break;

            case "enum": {
                editor = this.createChild(definition.ui_type == "radio" ? Radio : Dropdown, (e) => {
                    e.label = definition.name;
                    e.choices = definition.choices ?? {"": ""};
                    e.value = definition.default ?? null;
                });
            } break;

            case "color": {
                editor = this.createChild(ColorPicker, (e) => {
                    e.label = definition.name;
                    e.channels = definition.channels ?? 3;
                    e.value = definition.default ?? "#000000";
                }, definition.channels);
            } break;

            case "image": {
                editor = this.createChild(ImageBox, (e) => {
                    e.label = definition.name;
                    e.channels = definition.channels ?? 3;
                });
            } break;

            // TODO
            case "seed": {
                editor = this.createChild(SeedBox, (e) => {
                    e.label = definition.name;
                });
            } break;

            default: {
                console.log(`WARNING: Unhandled type ${definition.type}`);
            } break;
        }

        if (editor) {
            editor.onValueChange.connect((value) => {
                this.onValueChange.fire(value);
            });
        }

        this.onValueChange = new Signal();
    }
}
customElements.define("configurable-param-editor", ConfigurableParamEditor);
