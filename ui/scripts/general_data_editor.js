import {Form} from "../scripts/base/form.js";
import {NumberBox} from "../scripts/base/number_box.js";
import {FieldManager} from "../scripts/core/field_manager.js";
import {Signal} from "../scripts/core/signal.js";
import {NoiseEditor} from "../scripts/noise_editor.js";
import {SeedBox} from "../scripts/seed_box.js";

export class GeneralDataEditor extends Form {
    constructor() {
        super();

        this.onValueChange = new Signal();

        this._manager = new FieldManager(this.onValueChange);

        this.createField("Initial noise", NoiseEditor, (e) => {
            this._manager.manage(e, "initial_noise");
        });

        this.createField("Parallel", NumberBox, (e) => {
            e.minimum = 1;
            e.step = 1;
            e.value = 1;
            this._manager.manage(e, "parallel");
        });

        this.createField("Seed", SeedBox, (e) => {
            this._manager.manage(e, "seed");
        });
    }

    get value() {
        return this._manager.value;
    }

    set value(value) {
        this._manager.value = value;
    }
}
customElements.define("general-data-editor", GeneralDataEditor);
