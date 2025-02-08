import {BoolEditor} from "../scripts/base/bool_editor.js";
import {Column} from "../scripts/base/column.js";
import {ImageEditor} from "../scripts/base/image_editor.js";
import {NumberEditor} from "../scripts/base/number_editor.js";
import {Row} from "../scripts/base/row.js";
import {Signal} from "../scripts/core/signal.js";
import {SeedEditor} from "../scripts/seed_editor.js";

export class NoiseEditor extends Row {
    constructor() {
        super();

        let noise = {};

        this.createChild(ImageEditor);

        this.createChild(Column, (e) => {
            e.createChild(NumberEditor, (e) => {
                e.label = "Scale";
                e.variant = "slider";
                e.minimum = 1;
                e.maximum = 1024;
                e.step = 1;
                e.value = 1;
                e.onValueChange.connect((value) => {
                    noise.scale = value;

                    this.onValueChange.fire(noise);
                });
            });

            e.createChild(NumberEditor, (e) => {
                e.label = "Detail";
                e.variant = "slider";
                e.minimum = 1.0;
                e.maximum = 10.0;
                e.step = 0.01;
                e.value = 1.0;
                e.onValueChange.connect((value) => {
                    noise.detail = value;

                    this.onValueChange.fire(noise);
                });
            });

            e.createChild(NumberEditor, (e) => {
                e.label = "Lacunarity";
                e.variant = "slider";
                e.minimum = 0.01;
                e.maximum = 4.0;
                e.step = 0.01;
                e.value = 2.0;
                e.onValueChange.connect((value) => {
                    noise.lacunarity = value;

                    this.onValueChange.fire(noise);
                });
            });

            e.createChild(NumberEditor, (e) => {
                e.label = "Persistence";
                e.variant = "slider";
                e.minimum = 0.0;
                e.maximum = 1.0;
                e.step = 0.01;
                e.value = 0.5;
                e.onValueChange.connect((value) => {
                    noise.persistence = value;

                    this.onValueChange.fire(noise);
                });
            });

            e.createChild(SeedEditor, (e) => {
                e.onValueChange.connect((value) => {
                    noise.seed = value;

                    this.onValueChange.fire(noise);
                });
            });

            e.createChild(BoolEditor, (e) => {
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
