import {Form} from "../scripts/base/form.js";
import {FieldManager} from "../scripts/core/field_manager.js";
import {Signal} from "../scripts/core/signal.js";
import {ConfigurableParamEditor} from "../scripts/configurable_param_editor.js";

export class OptionCategoryEditor extends Form {
    constructor(definition) {
        super();

        this.onValueChange = new Signal();

        this._manager = new FieldManager(this.onValueChange);
        this._manager.value = {__type__: definition.type};

        for (let [key, param] of Object.entries(definition.parameters)) {
            this.createField(param.name, ConfigurableParamEditor, (e) => {
                this._manager.manage(e, key);
            }, param);
        }
    }

    get value() {
        return this._manager.value;
    }

    set value(value) {
        this._manager.value = value;
    }
}
customElements.define("option-category-editor", OptionCategoryEditor);
