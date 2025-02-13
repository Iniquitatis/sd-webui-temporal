import {ColorPicker} from "../scripts/base/color_picker.js";
import {Column} from "../scripts/base/column.js";
import {ImageBox} from "../scripts/base/image_box.js";
import {NumberBox} from "../scripts/base/number_box.js";
import {Radio} from "../scripts/base/radio.js";
import {Row} from "../scripts/base/row.js";
import {FieldManager} from "../scripts/core/field_manager.js";
import {Signal} from "../scripts/core/signal.js";
import {postRequest} from "../scripts/utils/requests.js";

export class PatternEditor extends Row {
    constructor() {
        super();

        this.onValueChange = new Signal();

        this._manager = new FieldManager(this.onValueChange);

        this._preview = this.createChild(ImageBox, (e) => {
            e.label = "Preview";

            this.onValueChange.connect((value) => {
                postRequest("/temporal/render_texture", {
                    "type": "pattern",
                    "data": value,
                    "size": [256, 256],
                    "channels": 4,
                }, (result) => {
                    e.value = `data:image/png;base64,${result}`;
                });
            });
        });

        this.createChild(Column, (e) => {
            e.createChild(Radio, (e) => {
                e.label = "Type";
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

            e.createChild(NumberBox, (e) => {
                e.label = "Size";
                e.minimum = 1;
                e.step = 1;
                e.value = 8;
                this._manager.manage(e, "size");
            });

            e.createChild(ColorPicker, (e) => {
                e.label = "Color A";
                e.value = "#ffffffff";
                this._manager.manage(e, "color_a");
            }, 4);

            e.createChild(ColorPicker, (e) => {
                e.label = "Color B";
                e.value = "#000000ff";
                this._manager.manage(e, "color_b");
            }, 4);
        });
    }

    get value() {
        return this._manager.value;
    }

    set value(value) {
        this._manager.value = value;
    }
}
customElements.define("pattern-editor", PatternEditor);
