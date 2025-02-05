import {Column} from "../scripts/core/layout.js";
import {Signal} from "../scripts/core/signal.js";
import {NumberEditor} from "../scripts/core/value_editors.js";
import {NoiseEditor} from "../scripts/noise_editor.js";

export class InitialNoiseEditor extends Column {
    constructor() {
        super();

        let initialNoise = {};

        this.createChild(NumberEditor, (e) => {
            e.label = "Factor";
            e.variant = "slider";
            e.minimum = 0.0;
            e.maximum = 1.0;
            e.step = 0.01;
            e.value = 0.0;
            e.onValueChange.connect((value) => {
                initialNoise.factor = value

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
