import {Button} from "/scripts/base/button.js";
import {Column} from "/scripts/base/column.js";
import {Signal} from "/scripts/core/signal.js";
import {defineElement} from "/scripts/utils/dom.js";
import {ObjectForm} from "/scripts/object_form.js";

export class SettingsEditor extends Column {
    static tag = "ce-settings-editor";

    constructor() {
        super();

        this.onValueChange = new Signal();
        this.onApply = new Signal();

        this.createChild(Button, (e) => {
            e.label = "Apply";
            e.onClick.connect(() => {
                this.onApply.fire(this._editor.value);
            });
        });

        this._editor = this.createChild(ObjectForm, (e) => {
            e.manageAll();
            e.onValueChange.connect((value) => {
                this.onValueChange.fire(value);
            });
        }, "modules.settings.Settings");
    }

    get value() {
        return this._editor.value;
    }

    set value(value) {
        this._editor.value = value;
    }
}
defineElement(SettingsEditor);
