import {Form} from "../scripts/base/form.js";
import {FieldManager} from "../scripts/core/field_manager.js";
import {Signal} from "../scripts/core/signal.js";
import {ConfigurableParamEditor} from "../scripts/configurable_param_editor.js";

export class ConfigurableParamForm extends Form {
    constructor(params, manager = null) {
        super(false);

        this.onValueChange = manager ? manager.onValueChange : new Signal();

        this._manager = manager ?? new FieldManager(this.onValueChange);

        for (let [key, param] of Object.entries(params)) {
            this.createField(param.name, ConfigurableParamEditor, (e) => {
                this._manager.manage(e, key);

                if (!param.dependencies) return;

                this.onValueChange.connect((value) => {
                    for (let [depKey, depValue] of Object.entries(param.dependencies)) {
                        if (value[depKey] == depValue) continue;
                        e.formItem.visible = false;
                        return;
                    }

                    e.formItem.visible = true;
                });
            }, param);
        }
    }
}
customElements.define("configurable-param-form", ConfigurableParamForm);
