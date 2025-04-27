import {CanvasWidget, TOOLS} from "../../scripts/base/canvas_widget.js";
import {DockGroup} from "../../scripts/base/dock_group.js";
import {Form} from "../../scripts/base/form.js";
import {ImageWidget} from "../../scripts/base/image_widget.js";
import {MediaBox} from "../../scripts/base/media_box.js";
import {Overlay} from "../../scripts/base/overlay.js";
import {Radio} from "../../scripts/base/radio.js";
import {Slider} from "../../scripts/base/slider.js";
import {VectorEditor} from "../../scripts/base/vector_editor.js";
import {Signal} from "../../scripts/core/signal.js";
import {clearElement, createElement} from "../../scripts/utils/dom.js";
import {mapValues} from "../../scripts/utils/object.js";

class ImageEditor extends Overlay {
    constructor() {
        super();

        this.onAccept = new Signal();

        this._canvas = this.createChild(CanvasWidget);

        this.createChild(DockGroup, (e) => {
            e.style.textAlign = "initial";

            e.createDock("\u{f0b2}", "Parameters", Form, (e) => {
                e.createField("Size", VectorEditor, (e) => {
                    e.minimum = 64;
                    e.maximum = 2048;
                    e.step = 8;
                    e.value = {x: this._canvas.width, y: this._canvas.height};
                    e.onValueChange.connect((value) => {
                        this._canvas.width = value.x;
                        this._canvas.height = value.y;
                    });
                    this._canvas.onValueChange.connect((value) => {
                        e.onValueChange.withDisabled(() => {
                            e.value = {x: this._canvas.width, y: this._canvas.height};
                        });
                    });
                }, Slider, {x: "X", y: "Y"});
            });

            e.createDock("\u{f1fc}", "Drawing", Form, (e) => {
                e.createField("Tool", Radio, (e) => {
                    e.choices = mapValues(TOOLS, (key, tool) => `${tool.icon} ${tool.name}`);
                    e.value = "none";
                    e.onValueChange.connect((value) => {
                        this._canvas.tool = value;

                        clearElement(this._ui);

                        this._canvas.tool.makeUI(this._ui);
                    });
                });

                this._ui = e.createChild(Form);
            });
        });

        this.addTool("\u{f00c}", () => {
            this.onAccept.fire(this.value);

            this.close();
        });
    }

    get value() {
        return this._canvas.value;
    }

    set value(value) {
        this._canvas.value = value;
    }
}
customElements.define("image-editor", ImageEditor);

export class ImageBox extends MediaBox {
    constructor(features = []) {
        super(ImageWidget, "image/*", features);

        if (features.includes("edit")) {
            this.addTool("\u{f303}", () => {
                createElement(document.body, ImageEditor, (e) => {
                    e.value = this.value;
                    e.onAccept.connect((value) => {
                        this.value = value;
                    });
                });
            });
        }
    }
}
customElements.define("image-box", ImageBox);
