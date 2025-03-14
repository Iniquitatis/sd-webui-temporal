import {Block} from "../scripts/base/block.js";
import {FieldManager} from "../scripts/core/field_manager.js";
import {Signal} from "../scripts/core/signal.js";
import {ParamForm} from "../scripts/param_form.js";

export class OptionCategoryEditor extends Block {
    constructor(definition) {
        super();

        this.onValueChange = new Signal();

        this._manager = new FieldManager(this.onValueChange);
        this._manager.value = {__type__: definition.type};

        this.createChild(ParamForm, null, definition.parameters, this._manager);
    }

    get value() {
        return this._manager.value;
    }

    set value(value) {
        this._manager.value = value;
    }
}
customElements.define("option-category-editor", OptionCategoryEditor);
