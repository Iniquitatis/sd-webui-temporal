import {ColorPicker} from "../scripts/base/color_picker.js";
import {Column} from "../scripts/base/column.js";
import {ImageBox} from "../scripts/base/image_box.js";
import {NumberBox} from "../scripts/base/number_box.js";
import {Radio} from "../scripts/base/radio.js";
import {Row} from "../scripts/base/row.js";
import {FieldManager} from "../scripts/core/field_manager.js";
import {Signal} from "../scripts/core/signal.js";
import {postRequest} from "../scripts/utils/requests.js";

export class GradientEditor extends Row {
    constructor() {
        super();

        this.onValueChange = new Signal();

        this._manager = new FieldManager(this.onValueChange);

        this._preview = this.createChild(ImageBox, (e) => {
            e.label = "Preview";

            this.onValueChange.connect((value) => {
                postRequest("/temporal/render_texture", {
                    "type": "gradient",
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
                    "linear": "Linear",
                    "radial": "Radial",
                };
                e.value = "linear";
                this._manager.manage(e, "type");
            });

            e.createChild(Row, (e) => {
                e.createChild(NumberBox, (e) => {
                    e.label = "Start X";
                    e.step = 0.01;
                    e.value = 0.0;
                    this._manager.manage(e, "start_x");
                });

                e.createChild(NumberBox, (e) => {
                    e.label = "Start Y";
                    e.step = 0.01;
                    e.value = 0.0;
                    this._manager.manage(e, "start_y");
                });
            });

            e.createChild(Row, (e) => {
                e.createChild(NumberBox, (e) => {
                    e.label = "End X";
                    e.step = 0.01;
                    e.value = 1.0;
                    this._manager.manage(e, "end_x");
                });

                e.createChild(NumberBox, (e) => {
                    e.label = "End Y";
                    e.step = 0.01;
                    e.value = 1.0;
                    this._manager.manage(e, "end_y");
                });
            });

            e.createChild(ColorPicker, (e) => {
                e.label = "Start color";
                e.value = "#ffffffff";
                this._manager.manage(e, "start_color");
            }, 4);

            e.createChild(ColorPicker, (e) => {
                e.label = "End color";
                e.value = "#000000ff";
                this._manager.manage(e, "end_color");
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
customElements.define("gradient-editor", GradientEditor);
