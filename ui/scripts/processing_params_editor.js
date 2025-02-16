import {Dropdown} from "../scripts/base/dropdown.js";
import {Form} from "../scripts/base/form.js";
import {Slider} from "../scripts/base/slider.js";
import {TextArea} from "../scripts/base/text_area.js";
import {FieldManager} from "../scripts/core/field_manager.js";
import {Signal} from "../scripts/core/signal.js";
import {SeedBox} from "../scripts/seed_box.js";
import {models, samplers, schedulers, vaes} from "../scripts/shared_data.js";

export class ProcessingParamsEditor extends Form {
    constructor() {
        super();

        this.onValueChange = new Signal();

        this._manager = new FieldManager(this.onValueChange);

        this.createRow((e) => {
            e.createField("Model", Dropdown, (e) => {
                e.choices = models;
                this._manager.manage(e, "model");
            });

            e.createField("VAE", Dropdown, (e) => {
                e.choices = vaes;
                this._manager.manage(e, "vae");
            });
        });

        this.createField("CLIP skip", Slider, (e) => {
            e.minimum = 1;
            e.maximum = 12;
            e.step = 1;
            e.value = 1;
            this._manager.manage(e, "clip_skip");
        });

        this.createField("Positive prompt", TextArea, (e) => {
            this._manager.manage(e, "positive_prompt");
        });

        this.createField("Negative prompt", TextArea, (e) => {
            this._manager.manage(e, "negative_prompt");
        });

        this.createRow((e) => {
            e.createField("Sampler", Dropdown, (e) => {
                e.choices = samplers;
                this._manager.manage(e, "sampler");
            });

            e.createField("Scheduler", Dropdown, (e) => {
                e.choices = schedulers;
                this._manager.manage(e, "scheduler");
            });
        });

        this.createField("Steps", Slider, (e) => {
            e.minimum = 1;
            e.maximum = 150;
            e.step = 1;
            e.value = 20;
            this._manager.manage(e, "steps");
        });

        this.createField("CFG", Slider, (e) => {
            e.minimum = 1.0;
            e.maximum = 30.0;
            e.step = 0.5;
            e.value = 5.0;
            this._manager.manage(e, "cfg");
        });

        this.createField("Strength", Slider, (e) => {
            e.minimum = 0.0;
            e.maximum = 1.0;
            e.step = 0.01;
            e.value = 0.5;
            this._manager.manage(e, "strength");
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
customElements.define("processing-params-editor", ProcessingParamsEditor);
