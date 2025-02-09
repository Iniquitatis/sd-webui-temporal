import {Checkbox} from "../scripts/base/checkbox.js";
import {Column} from "../scripts/base/column.js";
import {ImageBox} from "../scripts/base/image_box.js";
import {Radio} from "../scripts/base/radio.js";
import {Slider} from "../scripts/base/slider.js";
import {Row} from "../scripts/base/row.js";
import {Signal} from "../scripts/core/signal.js";
import {SeedBox} from "../scripts/seed_box.js";

export class NoiseEditor extends Row {
    constructor() {
        super();

        let noise = {};

        this.createChild(ImageBox);

        this.createChild(Column, (e) => {
            e.createChild(Radio, (e) => {
                e.label = "Mode";
                e.choices = {
                    "fbm": "fBm",
                    "turbulence": "Turbulence",
                    "ridge": "Ridge",
                };
                e.onValueChange.connect((value) => {
                    noise.mode = value;

                    this.onValueChange.fire(noise);
                });
            });

            e.createChild(Slider, (e) => {
                e.label = "Scale";
                e.minimum = 1;
                e.maximum = 1024;
                e.step = 1;
                e.value = 1;
                e.onValueChange.connect((value) => {
                    noise.scale = value;

                    this.onValueChange.fire(noise);
                });
            });

            e.createChild(Slider, (e) => {
                e.label = "Detail";
                e.minimum = 1.0;
                e.maximum = 10.0;
                e.step = 0.01;
                e.value = 1.0;
                e.onValueChange.connect((value) => {
                    noise.detail = value;

                    this.onValueChange.fire(noise);
                });
            });

            e.createChild(Slider, (e) => {
                e.label = "Lacunarity";
                e.minimum = 0.01;
                e.maximum = 4.0;
                e.step = 0.01;
                e.value = 2.0;
                e.onValueChange.connect((value) => {
                    noise.lacunarity = value;

                    this.onValueChange.fire(noise);
                });
            });

            e.createChild(Slider, (e) => {
                e.label = "Persistence";
                e.minimum = 0.0;
                e.maximum = 1.0;
                e.step = 0.01;
                e.value = 0.5;
                e.onValueChange.connect((value) => {
                    noise.persistence = value;

                    this.onValueChange.fire(noise);
                });
            });

            e.createChild(SeedBox, (e) => {
                e.label = "Seed",
                e.onValueChange.connect((value) => {
                    noise.seed = value;

                    this.onValueChange.fire(noise);
                });
            });

            e.createChild(Checkbox, (e) => {
                e.label = "Use global seed";
                e.value = false;
                e.onValueChange.connect((value) => {
                    noise.use_global_seed = value;

                    this.onValueChange.fire(noise);
                });
            });
        });

        this.onValueChange = new Signal();
    }
}
customElements.define("noise-editor", NoiseEditor);
