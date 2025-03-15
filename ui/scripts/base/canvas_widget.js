import {Block} from "../../scripts/base/block.js";
import {Signal} from "../../scripts/core/signal.js";
import {colorToHex} from "../../scripts/utils/color.js";

export class CanvasWidget extends Block {
    constructor() {
        super();

        this.brushEnabled = false;
        this.brushColor = {r: 0.0, g: 0.0, b: 0.0, a: 1.0};
        this.brushThickness = 16;

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

    fill() {
        this._mainCtx.fillStyle = colorToHex(this.brushColor);
        this._mainCtx.fillRect(0, 0, this.width, this.height);

        this._value = this._mainCanvas.toDataURL("image/png");
        this.onValueChange.fire(this._value);
    }

    flipH() {
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
    }

    flipV() {
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
    }

    _onMouseDown(event) {
        if (!this.brushEnabled || event.button != 0) return;

        this._overlayCanvas.style.opacity = `${this.brushColor.a}`;

        this._overlayCtx.imageSmoothingEnabled = true;
        this._overlayCtx.lineCap = "round";
        this._overlayCtx.lineJoin = "round";
        this._overlayCtx.lineWidth = this.brushThickness;
        this._overlayCtx.strokeStyle = colorToHex(this.brushColor, 3);

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
        this._mainCtx.globalAlpha = this.brushColor.a;
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
