import {Form} from "../scripts/base/form.js";
import {NumberBox} from "../scripts/base/number_box.js";
import {TextArea} from "../scripts/base/text_area.js";
import {TextBox} from "../scripts/base/text_box.js";
import {VectorEditor} from "../scripts/base/vector_editor.js";
import {FieldManager} from "../scripts/core/field_manager.js";
import {Signal} from "../scripts/core/signal.js";
import {SeedBox} from "../scripts/seed_box.js";

export class GeneralDataEditor extends Form {
    constructor() {
        super();

        this.onValueChange = new Signal();
        this.onImageSizeChange = new Signal();

        this._manager = new FieldManager(this.onValueChange);

        this.createField("Name", TextBox, (e) => {
            this._manager.manage(e, "name");
        });

        this.createField("Description", TextArea, (e) => {
            this._manager.manage(e, "description");
        });

        this.createField("Image size", VectorEditor, (e) => {
            e.minimum = 64;
            e.maximum = 2048;
            e.step = 8;
            e.value = {x: 512, y: 512};
            e.onValueChange.connect((value) => {
                this.onImageSizeChange.fire(value);
            });
            this._manager.manage(e, "image_size");
        }, NumberBox, {x: "X", y: "Y"});

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
