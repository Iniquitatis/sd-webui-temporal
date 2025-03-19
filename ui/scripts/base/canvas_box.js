import {CanvasWidget, TOOLS} from "../../scripts/base/canvas_widget.js";
import {Form} from "../../scripts/base/form.js";
import {MediaBox} from "../../scripts/base/media_box.js";
import {Radio} from "../../scripts/base/radio.js";
import {clearElement} from "../../scripts/utils/dom.js";
import {mapObject} from "../../scripts/utils/object.js";

export class CanvasBox extends MediaBox {
    constructor(features = []) {
        super(CanvasWidget, "image/*", features);

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
    }

    set canvasWidth(value) {
        this._element.width = value;
    }
}
customElements.define("canvas-box", CanvasBox);
