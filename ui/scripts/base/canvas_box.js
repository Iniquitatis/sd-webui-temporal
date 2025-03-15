import {Button} from "../../scripts/base/button.js";
import {CanvasWidget} from "../../scripts/base/canvas_widget.js";
import {Checkbox} from "../../scripts/base/checkbox.js";
import {ColorPicker} from "../../scripts/base/color_picker.js";
import {Form} from "../../scripts/base/form.js";
import {MediaBox} from "../../scripts/base/media_box.js";
import {Slider} from "../../scripts/base/slider.js";

export class CanvasBox extends MediaBox {
    constructor(features = []) {
        super(CanvasWidget, "image/*", features);

        this.tools.createDock("\u{f1fc}", "Drawing", Form, (e) => {
            e.createField("Brush enabled", Checkbox, (e) => {
                e.value = this._element.brushEnabled;
                e.onValueChange.connect((value) => {
                    this._element.brushEnabled = value;
                });
            });

            e.createField("Brush color", ColorPicker, (e) => {
                e.value = this._element.brushColor;
                e.onValueChange.connect((value) => {
                    this._element.brushColor = value;
                });
            });

            e.createField("Brush thickness", Slider, (e) => {
                e.minimum = 1;
                e.maximum = 128;
                e.step = 1;
                e.value = this._element.brushThickness;
                e.onValueChange.connect((value) => {
                    this._element.brushThickness = value;
                });
            });

            e.createChild(Button, (e) => {
                e.label = "\u{f575} Fill";
                e.onClick.connect(() => {
                    this._element.fill();
                });
            });

            e.createChild(Button, (e) => {
                e.label = "\u{f0ec} Flip horizontally";
                e.onClick.connect(() => {
                    this._element.flipH();
                });
            });

            e.createChild(Button, (e) => {
                e.label = "\u{e099} Flip vertically";
                e.onClick.connect(() => {
                    this._element.flipV();
                });
            });
        });
    }
}
customElements.define("canvas-box", CanvasBox);
