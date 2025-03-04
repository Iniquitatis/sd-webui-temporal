import {Accordion} from "../scripts/base/accordion.js";
import {Button} from "../scripts/base/button.js";
import {Column} from "../scripts/base/column.js";
import {FieldManager} from "../scripts/core/field_manager.js";
import {Signal} from "../scripts/core/signal.js";
import {OptionCategoryEditor} from "../scripts/option_category_editor.js";
import {optionCategories} from "../scripts/shared_data.js";

export class SettingsEditor extends Column {
    constructor() {
        super();

        this.onValueChange = new Signal();
        this.onApply = new Signal();

        this._manager = new FieldManager(this.onValueChange);

        this.createChild(Button, (e) => {
            e.label = "Apply";
            e.onClick.connect(() => {
                this.onApply.fire(this._manager.value);
            });
        });

        for (let [key, category] of Object.entries(optionCategories)) {
            this.createChild(Accordion, (e) => {
                e.label = category.name;

                e.createChild(OptionCategoryEditor, (e) => {
                    this._manager.manage(e, key);
                }, category);
            });
        }
    }

    get value() {
        return this._manager.value;
    }

    set value(value) {
        this._manager.value = value;
    }
}
customElements.define("settings-editor", SettingsEditor);
