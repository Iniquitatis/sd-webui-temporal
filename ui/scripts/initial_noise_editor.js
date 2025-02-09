import {Column} from "../scripts/base/column.js";
import {Slider} from "../scripts/base/slider.js";
import {Signal} from "../scripts/core/signal.js";
import {NoiseEditor} from "../scripts/noise_editor.js";

export class InitialNoiseEditor extends Column {
    constructor() {
        super();

        let initialNoise = {};

        this.createChild(Slider, (e) => {
            e.label = "Factor";
            e.minimum = 0.0;
            e.maximum = 1.0;
            e.step = 0.01;
            e.value = 0.0;
            e.onValueChange.connect((value) => {
                initialNoise.factor = value;

                this.onValueChange.fire(initialNoise);
            });
        });

        this.createChild(NoiseEditor, (e) => {
            e.onValueChange.connect((value) => {
                initialNoise.noise = value;

                this.onValueChange.fire(initialNoise);
            });
        });

        this.onValueChange = new Signal();
    }
}
customElements.define("initial-noise-editor", InitialNoiseEditor);
