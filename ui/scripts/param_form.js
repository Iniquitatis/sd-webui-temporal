import {Form} from "../scripts/base/form.js";
import {FieldManager} from "../scripts/core/field_manager.js";
import {Signal} from "../scripts/core/signal.js";
import {ParamEditor} from "../scripts/param_editor.js";

export class ParamForm extends Form {
    constructor(params, manager = null) {
        super(false);

        this.onValueChange = manager ? manager.onValueChange : new Signal();

        this._manager = manager ?? new FieldManager(this.onValueChange);

        for (let [key, param] of Object.entries(params)) {
            let field = this.createField(param.name, ParamEditor, (e) => {
                this._manager.manage(e, key);

                if (!param.dependencies) return;

                this.onValueChange.connect((value) => {
                    toggleDependencies(value, e, param.dependencies);
                });
            }, param);

            // NOTE: Because field isn't yet appended to a form within its
            // initializer
            if (!param.dependencies) continue;

            toggleDependencies(createDefault(params), field, param.dependencies);
        }
    }
}
customElements.define("param-form", ParamForm);

function createDefault(params) {
    let result = {};

    for (let [key, param] of Object.entries(params)) {
        if (param.default !== undefined) {
            result[key] = param.default;
        }
    }

    return result;
}

function toggleDependencies(value, element, dependencies) {
    for (let [depKey, depValue] of Object.entries(dependencies)) {
        if (value[depKey] == depValue) continue;
        element.formItem.visible = false;
        return;
    }

    element.formItem.visible = true;
}
