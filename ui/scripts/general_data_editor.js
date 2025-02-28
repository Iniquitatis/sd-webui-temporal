import {CanvasBox} from "../scripts/base/canvas_box.js";
import {Form} from "../scripts/base/form.js";
import {NumberBox} from "../scripts/base/number_box.js";
import {Tabs} from "../scripts/base/tabs.js";
import {VectorEditor} from "../scripts/base/vector_editor.js";
import {FieldManager} from "../scripts/core/field_manager.js";
import {Signal} from "../scripts/core/signal.js";
import {NoiseEditor} from "../scripts/noise_editor.js";
import {SeedBox} from "../scripts/seed_box.js";

export class GeneralDataEditor extends Form {
    constructor() {
        super();

        this.onValueChange = new Signal();

        this._manager = new FieldManager(this.onValueChange);

        this.createChild(Tabs, (e) => {
            this._image = e.createTab("Image", CanvasBox, (e) => {
                e.height = "60vh";
                e._element.width = 512;
                e._element.height = 512;
                this._manager.manage(e, "image");
            });

            e.createTab("Initial noise", NoiseEditor, (e) => {
                this._manager.manage(e, "initial_noise");
            });
        });

        this.createField("Image size", VectorEditor, (e) => {
            e.minimum = 64;
            e.maximum = 2048;
            e.step = 8;
            e.value = {x: 512, y: 512};
            e.onValueChange.connect((value) => {
                this._image._element.width = value.x;
                this._image._element.height = value.y;
            });
            this._manager.manage(e, "image_size");
        }, NumberBox, {x: "X", y: "Y"});

        this.createField("Parallel", NumberBox, (e) => {
            e.minimum = 1;
            e.step = 1;
            e.value = 1;
            this._manager.manage(e, "parallel");
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
customElements.define("general-data-editor", GeneralDataEditor);
