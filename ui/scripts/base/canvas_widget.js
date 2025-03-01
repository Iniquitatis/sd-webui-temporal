import {Block} from "../../scripts/base/block.js";
import {Column} from "../../scripts/base/column.js";
import {Form} from "../../scripts/base/form.js";
import {Row} from "../../scripts/base/row.js";
import {Slider} from "../../scripts/base/slider.js";
import {ToolButton} from "../../scripts/base/tool_button.js";
import {Signal} from "../../scripts/core/signal.js";

export class CanvasWidget extends Block {
    constructor() {
        super();

        this.onValueChange = new Signal();

        this._value = null;

        this.style.alignItems = "center";
        this.style.display = "flex";
        this.style.inset = "0";
        this.style.justifyContent = "center";
        this.style.position = "absolute";
        this.style.userSelect = "none";

        this.createChild(Block, (e) => {
            e.style.height = "100%";
            e.style.position = "relative";

            this._mainCanvas = e.createChild("canvas", (e) => {
                e.height = 300;
                e.width = 400;
                e.style.height = "100%";
                e.style.maxHeight = "100%";
                e.style.maxWidth = "100%";

                this._mainCtx = e.getContext("2d");
            });

            this._overlayCanvas = e.createChild("canvas", (e) => {
                e.height = this.height;
                e.width = this.width;
                e.style.height = "100%";
                e.style.left = "0";
                e.style.maxHeight = "100%";
                e.style.maxWidth = "100%";
                e.style.opacity = "1.0";
                e.style.position = "absolute";
                e.style.top = "0";
                e.addEventListener("pointerdown", (event) => this._onMouseDown(event));

                this._overlayCtx = e.getContext("2d");
            });

            e.createChild(Column, (e) => {
                e.style.left = "var(--layout-padding)";
                e.style.pointerEvents = "none";
                e.style.position = "absolute";
                e.style.top = "var(--layout-padding)";
                e.style.width = "unset";

                e.style.gridColumnStart = "1";
                e.style.gridRowStart = "1";

                e.createChild(Row, (e) => {
                    e.createChild(ToolButton, (e) => {
                        e.label = "\u{f575}";
                        e.style.pointerEvents = "auto";
                        e.onClick.connect(() => {
                            this._mainCtx.clearRect(0, 0, this.width, this.height);

                            this._value = null;
                            this.onValueChange.fire(null);
                        });
                    });

                    e.createChild(ToolButton, (e) => {
                        e.label = "\u{f0ec}";
                        e.style.pointerEvents = "auto";
                        e.onClick.connect(() => {
                            this._overlayCtx.drawImage(this._mainCanvas, 0, 0);

                            this._mainCtx.clearRect(0, 0, this.width, this.height);
                            this._mainCtx.save();
                            this._mainCtx.translate(this.width, 0);
                            this._mainCtx.scale(-1.0, 1.0);
                            this._mainCtx.drawImage(this._overlayCanvas, 0, 0);
                            this._mainCtx.restore();

                            this._overlayCtx.clearRect(0, 0, this.width, this.height);

                            this._value = this._mainCanvas.toDataURL("image/png");
                            this.onValueChange.fire(this._value);
                        });
                    });

                    e.createChild(ToolButton, (e) => {
                        e.label = "\u{e099}";
                        e.style.pointerEvents = "auto";
                        e.onClick.connect(() => {
                            this._overlayCtx.drawImage(this._mainCanvas, 0, 0);

                            this._mainCtx.clearRect(0, 0, this.width, this.height);
                            this._mainCtx.save();
                            this._mainCtx.translate(0, this.height);
                            this._mainCtx.scale(1.0, -1.0);
                            this._mainCtx.drawImage(this._overlayCanvas, 0, 0);
                            this._mainCtx.restore();

                            this._overlayCtx.clearRect(0, 0, this.width, this.height);

                            this._value = this._mainCanvas.toDataURL("image/png");
                            this.onValueChange.fire(this._value);
                        });
                    });

                    e.createChild(ToolButton, (e) => {
                        e.label = "\u{f1fc}";
                        e.style.pointerEvents = "auto";
                        e.onClick.connect(() => {
                            this._paintingColumn.style.display = this._paintingColumn.style.display == "flex" ? "none" : "flex";
                        });
                    });
                });

                this._paintingColumn = e.createChild(Form, (e) => {
                    e.style.display = "none";
                    e.style.maxWidth = "15rem";

                    this._brushColor = e.createField("Color", "input", (e) => {
                        e.type = "color";
                        e.style.pointerEvents = "auto";
                        e.style.width = "100%";
                    });

                    this._brushOpacity = e.createField("Opacity", Slider, (e) => {
                        e.minimum = 0.0;
                        e.maximum = 1.0;
                        e.step = 0.01;
                        e.value = 1.0;
                        e.style.pointerEvents = "auto";
                    });

                    this._brushThickness = e.createField("Thickness", Slider, (e) => {
                        e.minimum = 1;
                        e.maximum = 128;
                        e.step = 1;
                        e.value = 16;
                        e.style.pointerEvents = "auto";
                    });
                });
            });
        });
    }

    get height() {
        return this._mainCanvas.height;
    }

    get value() {
        return this._value;
    }

    get width() {
        return this._mainCanvas.width;
    }

    set height(value) {
        this._mainCanvas.height = value;
        this._overlayCanvas.height = value;
    }

    set value(value) {
        // FIXME: Should receive the correct value in the first place
        if (value && !value.startsWith("data:image/")) {
            value = `data:image/png;base64,${value}`;
        }

        if (value) {
            let image = new Image();
            image.addEventListener("load", () => {
                this.width = image.width;
                this.height = image.height;

                this._mainCtx.drawImage(image, 0, 0);

                this._value = value;
                this.onValueChange.fire(value);
            });
            image.src = value;
        } else {
            this._mainCtx.clearRect(0, 0, this.width, this.height);

            this._value = null;
            this.onValueChange.fire(null);
        }
    }

    set width(value) {
        this._mainCanvas.width = value;
        this._overlayCanvas.width = value;
    }

    _onMouseDown(event) {
        if (event.button != 0) return;

        this._overlayCanvas.style.opacity = `${this._brushOpacity.value}`;

        this._overlayCtx.imageSmoothingEnabled = true;
        this._overlayCtx.lineCap = "round";
        this._overlayCtx.lineJoin = "round";
        this._overlayCtx.lineWidth = this._brushThickness.value;
        this._overlayCtx.strokeStyle = this._brushColor.value;

        this._lastPosition = this._stroke(this._overlayCanvas, event);

        currentWidget = this;
    }

    _onMouseMove(event) {
        this._lastPosition = this._stroke(
            this._overlayCanvas,
            event,
            this._lastPosition,
        );
    }

    _onMouseUp(event) {
        this._mainCtx.save();
        this._mainCtx.globalAlpha = this._brushOpacity.value;
        this._mainCtx.imageSmoothingEnabled = true;
        this._mainCtx.drawImage(this._overlayCanvas, 0, 0);
        this._mainCtx.restore();

        this._overlayCtx.clearRect(0, 0, this.width, this.height);

        this._value = this._mainCanvas.toDataURL("image/png");
        this.onValueChange.fire(this._value);

        currentWidget = null;
    }

    _stroke(canvas, event, lastPosition = null) {
        let ctx = canvas.getContext("2d");

        let rect = canvas.getBoundingClientRect();
        let position = [
            (event.clientX - rect.x) * (canvas.width / rect.width),
            (event.clientY - rect.y) * (canvas.height / rect.height),
        ];

        ctx.beginPath();
        ctx.moveTo(...(lastPosition ?? position));
        ctx.lineTo(...position);
        ctx.stroke();

        return position;
    }
}
customElements.define("canvas-widget", CanvasWidget);

let currentWidget = null;

window.addEventListener("touchmove", (event) => {
    if (!currentWidget) return;

    event.stopPropagation();
    event.preventDefault();
}, {passive: false});
window.addEventListener("pointermove", (event) => currentWidget?._onMouseMove(event));
window.addEventListener("pointerup", (event) => currentWidget?._onMouseUp(event));
