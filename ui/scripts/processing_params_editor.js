import {Column} from "../scripts/base/column.js";
import {Dropdown} from "../scripts/base/dropdown.js";
import {ImageBox} from "../scripts/base/image_box.js";
import {Row} from "../scripts/base/row.js";
import {Slider} from "../scripts/base/slider.js";
import {TextArea} from "../scripts/base/text_area.js";
import {FieldManager} from "../scripts/core/field_manager.js";
import {Signal} from "../scripts/core/signal.js";
import {SeedBox} from "../scripts/seed_box.js";
import {models, samplers, schedulers, vaes} from "../scripts/test_data.js";

export class ProcessingParamsEditor extends Column {
    constructor() {
        super();

        this.onValueChange = new Signal();

        this._manager = new FieldManager(this.onValueChange);

        this.createChild(Column, (e) => {
            e.createChild(ImageBox, (e) => {
                e.label = "Image";
                this._manager.manage(e, "images", (value) => value[0], (value) => [value]);
            });

            e.createChild(Row, (e) => {
                e.createChild(Dropdown, (e) => {
                    e.label = "Model";
                    e.choices = models;
                    this._manager.manage(e, "model");
                });

                e.createChild(Dropdown, (e) => {
                    e.label = "VAE";
                    e.choices = vaes;
                    this._manager.manage(e, "vae");
                });
            });

            e.createChild(Slider, (e) => {
                e.label = "CLIP skip";
                e.minimum = 1;
                e.maximum = 12;
                e.step = 1;
                e.value = 1;
                this._manager.manage(e, "clip_skip");
            });

            e.createChild(TextArea, (e) => {
                e.label = "Positive prompt";
                this._manager.manage(e, "positive_prompts", (value) => value[0], (value) => [value]);
            });

            e.createChild(TextArea, (e) => {
                e.label = "Negative prompt";
                this._manager.manage(e, "negative_prompts", (value) => value[0], (value) => [value]);
            });

            e.createChild(Row, (e) => {
                e.createChild(Slider, (e) => {
                    e.label = "Width";
                    e.minimum = 64;
                    e.maximum = 2048;
                    e.step = 8;
                    e.value = 512;
                    this._manager.manage(e, "width");
                });

                e.createChild(Slider, (e) => {
                    e.label = "Height";
                    e.minimum = 64;
                    e.maximum = 2048;
                    e.step = 8;
                    e.value = 512;
                    this._manager.manage(e, "height");
                });
            });

            e.createChild(Row, (e) => {
                e.createChild(Dropdown, (e) => {
                    e.label = "Sampler";
                    e.choices = samplers;
                    this._manager.manage(e, "sampler");
                });

                e.createChild(Dropdown, (e) => {
                    e.label = "Scheduler";
                    e.choices = schedulers;
                    this._manager.manage(e, "scheduler");
                });
            });

            e.createChild(Slider, (e) => {
                e.label = "Steps";
                e.minimum = 1;
                e.maximum = 150;
                e.step = 1;
                e.value = 20;
                this._manager.manage(e, "steps");
            });

            e.createChild(Slider, (e) => {
                e.label = "CFG";
                e.minimum = 1.0;
                e.maximum = 30.0;
                e.step = 0.5;
                e.value = 5.0;
                this._manager.manage(e, "cfg");
            });

            e.createChild(Slider, (e) => {
                e.label = "Strength";
                e.minimum = 0.0;
                e.maximum = 1.0;
                e.step = 0.01;
                e.value = 0.5;
                this._manager.manage(e, "strength");
            });

            e.createChild(SeedBox, (e) => {
                e.label = "Seed";
                this._manager.manage(e, "seeds", (value) => value[0], (value) => [value]);
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
customElements.define("processing-params-editor", ProcessingParamsEditor);
