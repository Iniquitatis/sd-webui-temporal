import {Form} from "../scripts/base/form.js";
import {Slider} from "../scripts/base/slider.js";
import {FieldManager} from "../scripts/core/field_manager.js";
import {Signal} from "../scripts/core/signal.js";
import {NoiseEditor} from "../scripts/noise_editor.js";

export class InitialNoiseEditor extends Form {
    constructor() {
        super();

        this.onValueChange = new Signal();

        this._manager = new FieldManager(this.onValueChange);

        this.createField("Factor", Slider, (e) => {
            e.minimum = 0.0;
            e.maximum = 1.0;
            e.step = 0.01;
            e.value = 0.0;
            this._manager.manage(e, "factor");
        });

        this.createField("Noise", NoiseEditor, (e) => {
            this._manager.manage(e, "noise");
        });
    }

    get value() {
        return this._manager.value;
    }

    set value(value) {
        this._manager.value = value;
    }
}
customElements.define("initial-noise-editor", InitialNoiseEditor);
