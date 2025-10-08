import {Button} from "/scripts/base/button.js";
import {CANVAS_TOOLS, CanvasWidget} from "/scripts/base/canvas_widget.js";
import {DockGroup} from "/scripts/base/dock_group.js";
import {Form} from "/scripts/base/form.js";
import {ImageWidget} from "/scripts/base/image_widget.js";
import {MediaBox} from "/scripts/base/media_box.js";
import {Overlay} from "/scripts/base/overlay.js";
import {Radio} from "/scripts/base/radio.js";
import {Signal} from "/scripts/core/signal.js";
import {clearElement, createElement, defineElement} from "/scripts/utils/dom.js";
import {mapValues} from "/scripts/utils/object.js";

class ImageEditor extends Overlay {
    static tag = "ce-image-editor";
    static css = `
        <self> > ce-dock-group {
            text-align: initial;
        }
    `;

    constructor() {
        super();

        this.onAccept = new Signal();

        this._canvas = this.createChild(CanvasWidget);

        this.createChild(DockGroup, (e) => {
            e.createDock("\u{f1fc}", "Tools", Form, (e) => {
                e.createField("Tool", Radio, (e) => {
                    e.choices = mapValues(CANVAS_TOOLS, (key, tool) => `${tool.icon} ${tool.name}`);
                    e.value = "none";
                    e.onValueChange.connect((value) => {
                        this._canvas.tool = value;

                        clearElement(this._ui);

                        this._canvas.tool.makeUI(this._ui);

                        if (this._canvas.tool.constructor.retained) {
                            this._ui.createChild(Button, (e) => {
                                e.label = "Apply";
                                e.onClick.connect(async () => {
                                    e.enabled = false;

                                    await this._canvas.applyRetainedTool();

                                    e.enabled = true;
                                });
                            });
                        }
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
defineElement(ImageEditor);

export class ImageBox extends MediaBox {
    static tag = "ce-image-box";

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
defineElement(ImageBox);
