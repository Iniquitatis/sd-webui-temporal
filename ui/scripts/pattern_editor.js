import {ColorPicker} from "../scripts/base/color_picker.js";
import {Form} from "../scripts/base/form.js";
import {NumberBox} from "../scripts/base/number_box.js";
import {Radio} from "../scripts/base/radio.js";
import {FieldManager} from "../scripts/core/field_manager.js";
import {Signal} from "../scripts/core/signal.js";

export class PatternEditor extends Form {
    constructor() {
        super();

        this.onValueChange = new Signal();

        this._manager = new FieldManager(this.onValueChange);

        this.createField("Type", Radio, (e) => {
            e.choices = {
                "horizontal_lines": "Horizontal lines",
                "vertical_lines": "Vertical lines",
                "diagonal_lines_nw": "Diagonal lines NW",
                "diagonal_lines_ne": "Diagonal lines NE",
                "checkerboard": "Checkerboard",
            };
            e.value = "horizontal_lines";
            this._manager.manage(e, "type");
        });

        this.createField("Size", NumberBox, (e) => {
            e.minimum = 1;
            e.step = 1;
            e.value = 8;
            this._manager.manage(e, "size");
        });

        this.createField("Color A", ColorPicker, (e) => {
            e.value = {r: 1.0, g: 1.0, b: 1.0, a: 1.0};
            this._manager.manage(e, "color_a");
        }, 4);

        this.createField("Color B", ColorPicker, (e) => {
            e.value = {r: 0.0, g: 0.0, b: 0.0, a: 1.0};
            this._manager.manage(e, "color_b");
        }, 4);
    }

    get value() {
        return this._manager.value;
    }

    set value(value) {
        this._manager.value = value;
    }
}
customElements.define("pattern-editor", PatternEditor);
