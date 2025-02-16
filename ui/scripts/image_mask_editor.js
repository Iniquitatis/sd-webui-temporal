import {Checkbox} from "../scripts/base/checkbox.js";
import {Form} from "../scripts/base/form.js";
import {ImageBox} from "../scripts/base/image_box.js";
import {Slider} from "../scripts/base/slider.js";
import {FieldManager} from "../scripts/core/field_manager.js";
import {Signal} from "../scripts/core/signal.js";

export class ImageMaskEditor extends Form {
    constructor() {
        super();

        this.onValueChange = new Signal();

        this._manager = new FieldManager(this.onValueChange);

        this.createField("Image", ImageBox, (e) => {
            this._manager.manage(e, "image");
        });

        this.createField("Normalized", Checkbox, (e) => {
            e.value = false;
            this._manager.manage(e, "normalized");
        });

        this.createField("Inverted", Checkbox, (e) => {
            e.value = false;
            this._manager.manage(e, "inverted");
        });

        this.createField("Blurring", Slider, (e) => {
            e.minimum = 0.0;
            e.maximum = 50.0;
            e.step = 0.1;
            e.value = 0.0;
            this._manager.manage(e, "blurring");
        });
    }

    get value() {
        return this._manager.value;
    }

    set value(value) {
        this._manager.value = value;
    }
}
customElements.define("image-mask-editor", ImageMaskEditor);
