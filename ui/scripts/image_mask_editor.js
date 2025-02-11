import {Checkbox} from "../scripts/base/checkbox.js";
import {Column} from "../scripts/base/column.js";
import {ImageBox} from "../scripts/base/image_box.js";
import {Slider} from "../scripts/base/slider.js";
import {FieldManager} from "../scripts/core/field_manager.js";
import {Signal} from "../scripts/core/signal.js";

export class ImageMaskEditor extends Column {
    constructor() {
        super();

        this.onValueChange = new Signal();

        this._manager = new FieldManager(this.onValueChange);

        this.createChild(ImageBox, (e) => {
            e.label = "Image";
            this._manager.manage(e, "image");
        });

        this.createChild(Checkbox, (e) => {
            e.label = "Normalized";
            e.value = false;
            this._manager.manage(e, "normalized");
        });

        this.createChild(Checkbox, (e) => {
            e.label = "Inverted";
            e.value = false;
            this._manager.manage(e, "inverted");
        });

        this.createChild(Slider, (e) => {
            e.label = "Blurring";
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
