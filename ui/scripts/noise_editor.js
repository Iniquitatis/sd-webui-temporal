import {Checkbox} from "../scripts/base/checkbox.js";
import {Form} from "../scripts/base/form.js";
import {Radio} from "../scripts/base/radio.js";
import {Slider} from "../scripts/base/slider.js";
import {FieldManager} from "../scripts/core/field_manager.js";
import {Signal} from "../scripts/core/signal.js";
import {SeedBox} from "../scripts/seed_box.js";

export class NoiseEditor extends Form {
    constructor() {
        super();

        this.onValueChange = new Signal();

        this._manager = new FieldManager(this.onValueChange);

        this.createField("Mode", Radio, (e) => {
            e.choices = {
                "fbm": "fBm",
                "turbulence": "Turbulence",
                "ridge": "Ridge",
            };
            e.value = "fbm";
            this._manager.manage(e, "mode");
        });

        this.createField("Scale", Slider, (e) => {
            e.minimum = 1;
            e.maximum = 1024;
            e.step = 1;
            e.value = 1;
            this._manager.manage(e, "scale");
        });

        this.createField("Detail", Slider, (e) => {
            e.minimum = 1.0;
            e.maximum = 10.0;
            e.step = 0.01;
            e.value = 1.0;
            this._manager.manage(e, "detail");
        });

        this.createField("Lacunarity", Slider, (e) => {
            e.minimum = 0.01;
            e.maximum = 4.0;
            e.step = 0.01;
            e.value = 2.0;
            this._manager.manage(e, "lacunarity");
        });

        this.createField("Persistence", Slider, (e) => {
            e.minimum = 0.0;
            e.maximum = 1.0;
            e.step = 0.01;
            e.value = 0.5;
            this._manager.manage(e, "persistence");
        });

        this.createField("Seed", SeedBox, (e) => {
            this._manager.manage(e, "seed");
        });

        this.createField("Use global seed", Checkbox, (e) => {
            e.value = false;
            this._manager.manage(e, "use_global_seed");
        });
    }

    get value() {
        return this._manager.value;
    }

    set value(value) {
        this._manager.value = value;
    }

    isComplexWidget() {
        return true;
    }
}
customElements.define("noise-editor", NoiseEditor);
