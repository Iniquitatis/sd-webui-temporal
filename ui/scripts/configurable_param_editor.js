import {BoolEditor} from "../scripts/base/bool_editor.js";
import {ColorEditor} from "../scripts/base/color_editor.js";
import {EnumEditor} from "../scripts/base/enum_editor.js";
import {ImageEditor} from "../scripts/base/image_editor.js";
import {NumberEditor} from "../scripts/base/number_editor.js";
import {TextEditor} from "../scripts/base/text_editor.js";
import {Signal} from "../scripts/core/signal.js";
import {Widget} from "../scripts/core/widget.js";
import {SeedEditor} from "../scripts/seed_editor.js";

export class ConfigurableParamEditor extends Widget {
    constructor(definition) {
        super();

        let editor = null;

        switch (definition.type) {
            case "bool": {
                editor = this.createChild(BoolEditor, (e) => {
                    e.label = definition.name;
                    e.value = definition.value;
                });
            } break;

            case "int": {
                editor = this.createChild(NumberEditor, (e) => {
                    e.label = definition.name;
                    e.variant = definition.ui_type ?? "box";
                    e.minimum = definition.minimum ?? undefined;
                    e.maximum = definition.maximum ?? undefined;
                    e.step = definition.step ?? 1;
                    e.value = definition.default ?? e.minimum;
                });
            } break;

            case "float": {
                editor = this.createChild(NumberEditor, (e) => {
                    e.label = definition.name;
                    e.variant = definition.ui_type ?? "box";
                    e.minimum = definition.minimum ?? undefined;
                    e.maximum = definition.maximum ?? undefined;
                    e.step = definition.step ?? 0.1;
                    e.value = definition.default ?? e.minimum;
                });
            } break;

            case "string": {
                editor = this.createChild(TextEditor, (e) => {
                    e.label = definition.name;
                    e.variant = definition.ui_type ?? "box";
                    e.value = definition.default ?? "";
                });
            } break;

            case "enum": {
                editor = this.createChild(EnumEditor, (e) => {
                    e.label = definition.name;
                    e.variant = definition.ui_type ?? "menu";
                    e.choices = definition.choices ?? {"": ""};
                    e.value = definition.default ?? null;
                });
            } break;

            case "color": {
                editor = this.createChild(ColorEditor, (e) => {
                    e.label = definition.name;
                    e.channels = definition.channels ?? 3;
                    e.value = definition.default ?? "#000000";
                }, definition.channels);
            } break;

            case "image": {
                editor = this.createChild(ImageEditor, (e) => {
                    e.label = definition.name;
                    e.channels = definition.channels ?? 3;
                });
            } break;

            // TODO
            case "seed": {
                editor = this.createChild(SeedEditor, (e) => {
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
