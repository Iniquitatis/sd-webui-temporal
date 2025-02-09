import {Column} from "../scripts/base/column.js";
import {EnumEditor} from "../scripts/base/enum_editor.js";
import {ImageEditor} from "../scripts/base/image_editor.js";
import {NumberEditor} from "../scripts/base/number_editor.js";
import {Row} from "../scripts/base/row.js";
import {TextEditor} from "../scripts/base/text_editor.js";
import {Signal} from "../scripts/core/signal.js";
import {SeedEditor} from "../scripts/seed_editor.js";
import {models, samplers, schedulers, vaes} from "../scripts/test_data.js";

export class ProcessingParamsEditor extends Column {
    constructor() {
        super();

        let parameters = {};

        this.createChild(ImageEditor, (e) => {
            e.label = "Image";
            e.onValueChange.connect((value) => {
                parameters.images = [value];

                this.onValueChange.fire(parameters);
            });
        });

        this.createChild(Row, (e) => {
            e.createChild(EnumEditor, (e) => {
                e.label = "Model";
                e.variant = "menu";
                e.choices = models;
                e.onValueChange.connect((value) => {
                    parameters.model = value;

                    this.onValueChange.fire(parameters);
                });
            });

            e.createChild(EnumEditor, (e) => {
                e.label = "VAE";
                e.variant = "menu";
                e.choices = vaes;
                e.onValueChange.connect((value) => {
                    parameters.vae = value;

                    this.onValueChange.fire(parameters);
                });
            });
        });

        this.createChild(NumberEditor, (e) => {
            e.label = "CLIP skip";
            e.variant = "slider";
            e.minimum = 1;
            e.maximum = 12;
            e.step = 1;
            e.value = 1;
            e.onValueChange.connect((value) => {
                parameters.clip_skip = value;

                this.onValueChange.fire(parameters);
            });
        });

        this.createChild(TextEditor, (e) => {
            e.label = "Positive prompt";
            e.variant = "area";
            e.onValueChange.connect((value) => {
                parameters.positive_prompts = [value];

                this.onValueChange.fire(parameters);
            });
        });

        this.createChild(TextEditor, (e) => {
            e.label = "Negative prompt";
            e.variant = "area";
            e.onValueChange.connect((value) => {
                parameters.negative_prompts = [value];

                this.onValueChange.fire(parameters);
            });
        });

        this.createChild(Row, (e) => {
            e.createChild(NumberEditor, (e) => {
                e.label = "Width";
                e.variant = "slider";
                e.minimum = 64;
                e.maximum = 2048;
                e.step = 8;
                e.value = 512;
                e.onValueChange.connect((value) => {
                    parameters.width = value;

                    this.onValueChange.fire(parameters);
                });
            });

            e.createChild(NumberEditor, (e) => {
                e.label = "Height";
                e.variant = "slider";
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
            e.createChild(EnumEditor, (e) => {
                e.label = "Sampler";
                e.variant = "menu";
                e.choices = samplers;
                e.onValueChange.connect((value) => {
                    parameters.sampler = value;

                    this.onValueChange.fire(parameters);
                });
            });

            e.createChild(EnumEditor, (e) => {
                e.label = "Scheduler";
                e.variant = "menu";
                e.choices = schedulers;
                e.onValueChange.connect((value) => {
                    parameters.scheduler = value;

                    this.onValueChange.fire(parameters);
                });
            });
        });

        this.createChild(NumberEditor, (e) => {
            e.label = "Steps";
            e.variant = "slider";
            e.minimum = 1;
            e.maximum = 150;
            e.step = 1;
            e.value = 20;
            e.onValueChange.connect((value) => {
                parameters.steps = value;

                this.onValueChange.fire(parameters);
            });
        });

        this.createChild(NumberEditor, (e) => {
            e.label = "CFG";
            e.variant = "slider";
            e.minimum = 1.0;
            e.maximum = 30.0;
            e.step = 0.5;
            e.value = 5.0;
            e.onValueChange.connect((value) => {
                parameters.cfg = value;

                this.onValueChange.fire(parameters);
            });
        });

        this.createChild(NumberEditor, (e) => {
            e.label = "Strength";
            e.variant = "slider";
            e.minimum = 0.0;
            e.maximum = 1.0;
            e.step = 0.01;
            e.value = 0.5;
            e.onValueChange.connect((value) => {
                parameters.strength = value;

                this.onValueChange.fire(parameters);
            });
        });

        this.createChild(SeedEditor, (e) => {
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
