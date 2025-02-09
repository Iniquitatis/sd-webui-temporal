import {Column} from "../scripts/base/column.js";
import {Dropdown} from "../scripts/base/dropdown.js";
import {ImageBox} from "../scripts/base/image_box.js";
import {Row} from "../scripts/base/row.js";
import {Slider} from "../scripts/base/slider.js";
import {TextArea} from "../scripts/base/text_area.js";
import {Signal} from "../scripts/core/signal.js";
import {SeedBox} from "../scripts/seed_box.js";
import {models, samplers, schedulers, vaes} from "../scripts/test_data.js";

export class ProcessingParamsEditor extends Column {
    constructor() {
        super();

        let parameters = {};

        this.createChild(ImageBox, (e) => {
            e.label = "Image";
            e.onValueChange.connect((value) => {
                parameters.images = [value];

                this.onValueChange.fire(parameters);
            });
        });

        this.createChild(Row, (e) => {
            e.createChild(Dropdown, (e) => {
                e.label = "Model";
                e.choices = models;
                e.onValueChange.connect((value) => {
                    parameters.model = value;

                    this.onValueChange.fire(parameters);
                });
            });

            e.createChild(Dropdown, (e) => {
                e.label = "VAE";
                e.choices = vaes;
                e.onValueChange.connect((value) => {
                    parameters.vae = value;

                    this.onValueChange.fire(parameters);
                });
            });
        });

        this.createChild(Slider, (e) => {
            e.label = "CLIP skip";
            e.minimum = 1;
            e.maximum = 12;
            e.step = 1;
            e.value = 1;
            e.onValueChange.connect((value) => {
                parameters.clip_skip = value;

                this.onValueChange.fire(parameters);
            });
        });

        this.createChild(TextArea, (e) => {
            e.label = "Positive prompt";
            e.onValueChange.connect((value) => {
                parameters.positive_prompts = [value];

                this.onValueChange.fire(parameters);
            });
        });

        this.createChild(TextArea, (e) => {
            e.label = "Negative prompt";
            e.onValueChange.connect((value) => {
                parameters.negative_prompts = [value];

                this.onValueChange.fire(parameters);
            });
        });

        this.createChild(Row, (e) => {
            e.createChild(Slider, (e) => {
                e.label = "Width";
                e.minimum = 64;
                e.maximum = 2048;
                e.step = 8;
                e.value = 512;
                e.onValueChange.connect((value) => {
                    parameters.width = value;

                    this.onValueChange.fire(parameters);
                });
            });

            e.createChild(Slider, (e) => {
                e.label = "Height";
                e.minimum = 64;
                e.maximum = 2048;
                e.step = 8;
                e.value = 512;
                e.onValueChange.connect((value) => {
                    parameters.height = value;

                    this.onValueChange.fire(parameters);
                });
            });
        });

        this.createChild(Row, (e) => {
            e.createChild(Dropdown, (e) => {
                e.label = "Sampler";
                e.choices = samplers;
                e.onValueChange.connect((value) => {
                    parameters.sampler = value;

                    this.onValueChange.fire(parameters);
                });
            });

            e.createChild(Dropdown, (e) => {
                e.label = "Scheduler";
                e.choices = schedulers;
                e.onValueChange.connect((value) => {
                    parameters.scheduler = value;

                    this.onValueChange.fire(parameters);
                });
            });
        });

        this.createChild(Slider, (e) => {
            e.label = "Steps";
            e.minimum = 1;
            e.maximum = 150;
            e.step = 1;
            e.value = 20;
            e.onValueChange.connect((value) => {
                parameters.steps = value;

                this.onValueChange.fire(parameters);
            });
        });

        this.createChild(Slider, (e) => {
            e.label = "CFG";
            e.minimum = 1.0;
            e.maximum = 30.0;
            e.step = 0.5;
            e.value = 5.0;
            e.onValueChange.connect((value) => {
                parameters.cfg = value;

                this.onValueChange.fire(parameters);
            });
        });

        this.createChild(Slider, (e) => {
            e.label = "Strength";
            e.minimum = 0.0;
            e.maximum = 1.0;
            e.step = 0.01;
            e.value = 0.5;
            e.onValueChange.connect((value) => {
                parameters.strength = value;

                this.onValueChange.fire(parameters);
            });
        });

        this.createChild(SeedBox, (e) => {
            e.label = "Seed";
            e.onValueChange.connect((value) => {
                parameters.seeds = [value];

                this.onValueChange.fire(parameters);
            });
        });

        this.onValueChange = new Signal();
    }
}
customElements.define("processing-params-editor", ProcessingParamsEditor);
