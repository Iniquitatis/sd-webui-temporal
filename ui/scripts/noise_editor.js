import {Checkbox} from "../scripts/base/checkbox.js";
import {Column} from "../scripts/base/column.js";
import {ImageBox} from "../scripts/base/image_box.js";
import {Radio} from "../scripts/base/radio.js";
import {Slider} from "../scripts/base/slider.js";
import {Row} from "../scripts/base/row.js";
import {FieldManager} from "../scripts/core/field_manager.js";
import {Signal} from "../scripts/core/signal.js";
import {postRequest} from "../scripts/utils/requests.js";
import {SeedBox} from "../scripts/seed_box.js";

export class NoiseEditor extends Row {
    constructor() {
        super();

        this.onValueChange = new Signal();

        this._manager = new FieldManager(this.onValueChange);

        this._preview = this.createChild(ImageBox, (e) => {
            e.label = "Preview";

            this.onValueChange.connect((value) => {
                postRequest("/temporal/render_preview", {
                    "type": "noise",
                    "data": value,
                    "size": [256, 256],
                    "channels": 3,
                }, (result) => {
                    e.value = `data:image/png;base64,${result}`;
                });
            });
        });

        this.createChild(Column, (e) => {
            e.createChild(Radio, (e) => {
                e.label = "Mode";
                e.choices = {
                    "fbm": "fBm",
                    "turbulence": "Turbulence",
                    "ridge": "Ridge",
                };
                e.value = "fbm";
                this._manager.manage(e, "mode");
            });

            e.createChild(Slider, (e) => {
                e.label = "Scale";
                e.minimum = 1;
                e.maximum = 1024;
                e.step = 1;
                e.value = 1;
                this._manager.manage(e, "scale");
            });

            e.createChild(Slider, (e) => {
                e.label = "Detail";
                e.minimum = 1.0;
                e.maximum = 10.0;
                e.step = 0.01;
                e.value = 1.0;
                this._manager.manage(e, "detail");
            });

            e.createChild(Slider, (e) => {
                e.label = "Lacunarity";
                e.minimum = 0.01;
                e.maximum = 4.0;
                e.step = 0.01;
                e.value = 2.0;
                this._manager.manage(e, "lacunarity");
            });

            e.createChild(Slider, (e) => {
                e.label = "Persistence";
                e.minimum = 0.0;
                e.maximum = 1.0;
                e.step = 0.01;
                e.value = 0.5;
                this._manager.manage(e, "persistence");
            });

            e.createChild(SeedBox, (e) => {
                e.label = "Seed",
                this._manager.manage(e, "seed");
            });

            e.createChild(Checkbox, (e) => {
                e.label = "Use global seed";
                e.value = false;
                this._manager.manage(e, "use_global_seed");
            });
        });
    }

    get value() {
        return this._manager.value;
    }

    set value(value) {
        this._manager.value = value;
    }
}
customElements.define("noise-editor", NoiseEditor);
