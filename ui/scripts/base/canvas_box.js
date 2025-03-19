import {CanvasWidget, TOOLS} from "../../scripts/base/canvas_widget.js";
import {Form} from "../../scripts/base/form.js";
import {MediaBox} from "../../scripts/base/media_box.js";
import {Radio} from "../../scripts/base/radio.js";
import {Slider} from "../../scripts/base/slider.js";
import {VectorEditor} from "../../scripts/base/vector_editor.js";
import {Signal} from "../../scripts/core/signal.js";
import {clearElement} from "../../scripts/utils/dom.js";
import {mapObject} from "../../scripts/utils/object.js";

export class CanvasBox extends MediaBox {
    constructor(features = []) {
        super(CanvasWidget, "image/*", features);

        this.onCanvasSizeChange = new Signal();

        this.tools.createDock("\u{f0b2}", "Parameters", Form, (e) => {
            e.createField("Size", VectorEditor, (e) => {
                e.minimum = 64;
                e.maximum = 2048;
                e.step = 8;
                e.value = {x: this.canvasWidth, y: this.canvasHeight};
                e.onValueChange.connect((value) => {
                    this.onCanvasSizeChange.withDisabled(() => {
                        this.canvasWidth = value.x;
                        this.canvasHeight = value.y;
                    });
                });
                this.onCanvasSizeChange.connect((value) => {
                    e.onValueChange.withDisabled(() => {
                        e.value = {x: value[0], y: value[1]};
                    });
                });
                this.onValueChange.connect((value) => {
                    e.onValueChange.withDisabled(() => {
                        e.value = {x: this.canvasWidth, y: this.canvasHeight};
                    });
                });
            }, Slider, {x: "X", y: "Y"});
        });

        this.tools.createDock("\u{f1fc}", "Drawing", Form, (e) => {
            e.createField("Tool", Radio, (e) => {
                e.choices = mapObject(TOOLS, (key, tool) => `${tool.icon} ${tool.name}`);
                e.value = "none";
                e.onValueChange.connect((value) => {
                    this._element.tool = value;

                    clearElement(this._ui);

                    this._element.tool.makeUI(this._ui);
                });
            });

            this._ui = e.createChild(Form);
        });
    }

    get canvasHeight() {
        return this._element.height;
    }

    get canvasWidth() {
        return this._element.width;
    }

    set canvasHeight(value) {
        this._element.height = value;

        this.onCanvasSizeChange.fire([this._element.width, this._element.height]);
    }

    set canvasWidth(value) {
        this._element.width = value;

        this.onCanvasSizeChange.fire([this._element.width, this._element.height]);
    }
}
customElements.define("canvas-box", CanvasBox);
