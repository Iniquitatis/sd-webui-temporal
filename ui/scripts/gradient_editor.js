import {ColorPicker} from "../scripts/base/color_picker.js";
import {Form} from "../scripts/base/form.js";
import {NumberBox} from "../scripts/base/number_box.js";
import {Radio} from "../scripts/base/radio.js";
import {VectorEditor} from "../scripts/base/vector_editor.js";
import {FieldManager} from "../scripts/core/field_manager.js";
import {Signal} from "../scripts/core/signal.js";

export class GradientEditor extends Form {
    constructor() {
        super();

        this.onValueChange = new Signal();

        this._manager = new FieldManager(this.onValueChange);

        this.createField("Type", Radio, (e) => {
            e.choices = {
                "linear": "Linear",
                "radial": "Radial",
            };
            e.value = "linear";
            this._manager.manage(e, "type");
        });

        this.createField("Start", VectorEditor, (e) => {
            e.step = 0.01;
            e.value = {x: 0.0, y: 0.0};
            this._manager.manage(e, "start");
        }, NumberBox, {x: "X", y: "Y"});

        this.createField("End", VectorEditor, (e) => {
            e.step = 0.01;
            e.value = {x: 1.0, y: 1.0};
            this._manager.manage(e, "end");
        }, NumberBox, {x: "X", y: "Y"});

        this.createField("Start color", ColorPicker, (e) => {
            e.value = {r: 1.0, g: 1.0, b: 1.0, a: 1.0};
            this._manager.manage(e, "start_color");
        }, 4);

        this.createField("End color", ColorPicker, (e) => {
            e.value = {r: 0.0, g: 0.0, b: 0.0, a: 1.0};
            this._manager.manage(e, "end_color");
        }, 4);
    }

    get value() {
        return this._manager.value;
    }

    set value(value) {
        this._manager.value = value;
    }
}
customElements.define("gradient-editor", GradientEditor);
