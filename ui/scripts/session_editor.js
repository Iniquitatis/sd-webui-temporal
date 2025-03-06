import {Checkbox} from "../scripts/base/checkbox.js";
import {Form} from "../scripts/base/form.js";
import {NumberBox} from "../scripts/base/number_box.js";
import {FieldManager} from "../scripts/core/field_manager.js";
import {Signal} from "../scripts/core/signal.js";

export class SessionEditor extends Form {
    constructor() {
        super();

        this.onValueChange = new Signal();

        this._manager = new FieldManager(this.onValueChange);

        this.createField("Load parameters", Checkbox, (e) => {
            e.value = true;
            e.onValueChange.connect((value) => {
                this._continue.visible = value;
            });
            this._manager.manage(e, "load_parameters");
        });

        this._continue = this.createField("Continue from last frame", Checkbox, (e) => {
            e.value = true;
            this._manager.manage(e, "continue_from_last_frame");
        });

        this.createField("Iteration count", NumberBox, (e) => {
            e.minimum = 1;
            e.step = 1;
            e.value = 10;
            this._manager.manage(e, "iter_count");
        });
    }

    get value() {
        return this._manager.value;
    }

    set value(value) {
        this._manager.value = value;
    }
}
customElements.define("session-editor", SessionEditor);
