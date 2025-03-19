import {Block} from "../../scripts/base/block.js";
import {Checkbox} from "../../scripts/base/checkbox.js";
import {ColorPicker} from "../../scripts/base/color_picker.js";
import {Slider} from "../../scripts/base/slider.js";
import {Signal} from "../../scripts/core/signal.js";
import {colorToHex} from "../../scripts/utils/color.js";

export const TOOLS = {};

class CanvasTool {
    constructor(mainCanvas, overlayCanvas) {
        this._mainCanvas = mainCanvas;
        this._mainCtx = mainCanvas.getContext("2d");
        this._overlayCanvas = overlayCanvas;
        this._overlayCtx = overlayCanvas.getContext("2d");
    }

    getEventMousePosition(canvas, event) {
        let rect = canvas.getBoundingClientRect();
        return [
            (event.clientX - rect.x) * (canvas.width / rect.width),
            (event.clientY - rect.y) * (canvas.height / rect.height),
        ];
    }

    makeUI(form) {}

    onMouseDown(event) {}

    onMouseMove(event) {}

    onMouseUp(event) {}
}

class NoneTool extends CanvasTool {
    static name = "None";
    static icon = "";
}
TOOLS.none = NoneTool;

class BrushTool extends CanvasTool {
    static name = "Brush";
    static icon = "\u{f1fc}";

    makeUI(form) {
        this._color = form.createField("Color", ColorPicker, (e) => {
            e.value = {r: 0.0, g: 0.0, b: 0.0, a: 1.0};
        });

        this._thickness = form.createField("Thickness", Slider, (e) => {
            e.minimum = 1;
            e.maximum = 128;
            e.step = 1;
            e.value = 16;
        });
    }

    onMouseDown(event) {
        if (event.button != 0) return;

        this._overlayCanvas.style.opacity = `${this._color.value.a}`;

        this._overlayCtx.lineCap = "round";
        this._overlayCtx.lineJoin = "round";
        this._overlayCtx.lineWidth = this._thickness.value;
        this._overlayCtx.strokeStyle = colorToHex(this._color.value, 3);

        this._lastPosition = this.getEventMousePosition(this._overlayCanvas, event);

        this.onMouseMove(event);
    }

    onMouseMove(event) {
        let position = this.getEventMousePosition(this._overlayCanvas, event);

        this._overlayCtx.beginPath();
        this._overlayCtx.moveTo(...this._lastPosition);
        this._overlayCtx.lineTo(...position);
        this._overlayCtx.stroke();

        this._lastPosition = position;
    }

    onMouseUp(event) {
        this._mainCtx.save();
        this._mainCtx.globalAlpha = this._color.value.a;
        this._mainCtx.drawImage(this._overlayCanvas, 0, 0);
        this._mainCtx.restore();

        this._overlayCtx.clearRect(0, 0, this._overlayCanvas.width, this._overlayCanvas.height);
    }
}
TOOLS.brush = BrushTool;

class LineTool extends CanvasTool {
    static name = "Line";
    static icon = "\u{f715}";

    makeUI(form) {
        this._color = form.createField("Color", ColorPicker, (e) => {
            e.value = {r: 0.0, g: 0.0, b: 0.0, a: 1.0};
        });

        this._thickness = form.createField("Thickness", Slider, (e) => {
            e.minimum = 1;
            e.maximum = 128;
            e.step = 1;
            e.value = 16;
        });
    }

    onMouseDown(event) {
        if (event.button != 0) return;

        this._overlayCanvas.style.opacity = `${this._color.value.a}`;

        this._overlayCtx.lineCap = "round";
        this._overlayCtx.lineJoin = "round";
        this._overlayCtx.lineWidth = this._thickness.value;
        this._overlayCtx.strokeStyle = colorToHex(this._color.value, 3);

        this._initialPosition = this.getEventMousePosition(this._overlayCanvas, event);

        this.onMouseMove(event);
    }

    onMouseMove(event) {
        this._overlayCtx.clearRect(0, 0, this._overlayCanvas.width, this._overlayCanvas.height);
        this._overlayCtx.beginPath();
        this._overlayCtx.moveTo(...this._initialPosition);
        this._overlayCtx.lineTo(...this.getEventMousePosition(this._overlayCanvas, event));
        this._overlayCtx.stroke();
    }

    onMouseUp(event) {
        this._mainCtx.save();
        this._mainCtx.globalAlpha = this._color.value.a;
        this._mainCtx.drawImage(this._overlayCanvas, 0, 0);
        this._mainCtx.restore();

        this._overlayCtx.clearRect(0, 0, this._overlayCanvas.width, this._overlayCanvas.height);
    }
}
TOOLS.line = LineTool;

class RectangleTool extends CanvasTool {
    static name = "Rectangle";
    static icon = "\u{f2fa}";

    makeUI(form) {
        this._color = form.createField("Color", ColorPicker, (e) => {
            e.value = {r: 0.0, g: 0.0, b: 0.0, a: 1.0};
        });

        this._thickness = form.createField("Thickness", Slider, (e) => {
            e.minimum = 1;
            e.maximum = 128;
            e.step = 1;
            e.value = 16;
        });

        this._filled = form.createField("Filled", Checkbox, (e) => {
            e.value = false;
        });
    }

    onMouseDown(event) {
        if (event.button != 0) return;

        this._overlayCanvas.style.opacity = `${this._color.value.a}`;

        if (this._filled.value) {
            this._overlayCtx.fillStyle = colorToHex(this._color.value, 3);
        } else {
            this._overlayCtx.lineCap = "butt";
            this._overlayCtx.lineJoin = "butt";
            this._overlayCtx.lineWidth = this._thickness.value;
            this._overlayCtx.strokeStyle = colorToHex(this._color.value, 3);
        }

        this._initialPosition = this.getEventMousePosition(this._overlayCanvas, event);

        this.onMouseMove(event);
    }

    onMouseMove(event) {
        let [ix, iy] = this._initialPosition;
        let [cx, cy] = this.getEventMousePosition(this._overlayCanvas, event);

        this._overlayCtx.clearRect(0, 0, this._overlayCanvas.width, this._overlayCanvas.height);
        this._overlayCtx.beginPath();
        this._overlayCtx.rect(ix, iy, cx - ix, cy - iy);

        if (this._filled.value) {
            this._overlayCtx.fill();
        } else {
            this._overlayCtx.stroke();
        }
    }

    onMouseUp(event) {
        this._mainCtx.save();
        this._mainCtx.globalAlpha = this._color.value.a;
        this._mainCtx.drawImage(this._overlayCanvas, 0, 0);
        this._mainCtx.restore();

        this._overlayCtx.clearRect(0, 0, this._overlayCanvas.width, this._overlayCanvas.height);
    }
}
TOOLS.rectangle = RectangleTool;

class EllipseTool extends CanvasTool {
    static name = "Ellipse";
    static icon = "\u{f111}";

    makeUI(form) {
        this._color = form.createField("Color", ColorPicker, (e) => {
            e.value = {r: 0.0, g: 0.0, b: 0.0, a: 1.0};
        });

        this._thickness = form.createField("Thickness", Slider, (e) => {
            e.minimum = 1;
            e.maximum = 128;
            e.step = 1;
            e.value = 16;
        });

        this._filled = form.createField("Filled", Checkbox, (e) => {
            e.value = false;
        });
    }

    onMouseDown(event) {
        if (event.button != 0) return;

        this._overlayCanvas.style.opacity = `${this._color.value.a}`;

        if (this._filled.value) {
            this._overlayCtx.fillStyle = colorToHex(this._color.value, 3);
        } else {
            this._overlayCtx.lineWidth = this._thickness.value;
            this._overlayCtx.strokeStyle = colorToHex(this._color.value, 3);
        }

        this._initialPosition = this.getEventMousePosition(this._overlayCanvas, event);

        this.onMouseMove(event);
    }

    onMouseMove(event) {
        let [ix, iy] = this._initialPosition;
        let [cx, cy] = this.getEventMousePosition(this._overlayCanvas, event);

        this._overlayCtx.clearRect(0, 0, this._overlayCanvas.width, this._overlayCanvas.height);
        this._overlayCtx.beginPath();
        this._overlayCtx.ellipse(
            (ix + cx) / 2,
            (iy + cy) / 2,
            Math.abs(cx - ix) / 2,
            Math.abs(cy - iy) / 2,
            0,
            0,
            Math.PI * 2.0,
        );

        if (this._filled.value) {
            this._overlayCtx.fill();
        } else {
            this._overlayCtx.stroke();
        }
    }

    onMouseUp(event) {
        this._mainCtx.save();
        this._mainCtx.globalAlpha = this._color.value.a;
        this._mainCtx.drawImage(this._overlayCanvas, 0, 0);
        this._mainCtx.restore();

        this._overlayCtx.clearRect(0, 0, this._overlayCanvas.width, this._overlayCanvas.height);
    }
}
TOOLS.ellipse = EllipseTool;

class FillTool extends CanvasTool {
    static name = "Fill";
    static icon = "\u{f575}";

    makeUI(form) {
        this._color = form.createField("Color", ColorPicker, (e) => {
            e.value = {r: 0.0, g: 0.0, b: 0.0, a: 1.0};
        });
    }

    onMouseDown(event) {
        if (event.button != 0) return;

        this._mainCtx.fillStyle = colorToHex(this._color.value);
        this._mainCtx.fillRect(0, 0, this._mainCanvas.width, this._mainCanvas.height);
    }
}
TOOLS.fill = FillTool;

class SmudgeTool extends CanvasTool {
    static name = "Smudge";
    static icon = "\u{e19e}";

    constructor(mainCanvas, overlayCanvas) {
        super(mainCanvas, overlayCanvas);

        this._maskCanvas = document.createElement("canvas");
        this._maskCtx = this._maskCanvas.getContext("2d");
    }

    makeUI(form) {
        this._strength = form.createField("Strength", Slider, (e) => {
            e.minimum = 0.0;
            e.maximum = 1.0;
            e.step = 0.01;
            e.value = 1.0;
        });

        this._hardness = form.createField("Hardness", Slider, (e) => {
            e.minimum = 0.0;
            e.maximum = 1.0;
            e.step = 0.01;
            e.value = 0.0;
        });

        this._thickness = form.createField("Thickness", Slider, (e) => {
            e.minimum = 1;
            e.maximum = 128;
            e.step = 1;
            e.value = 16;
        });
    }

    onMouseDown(event) {
        if (event.button != 0) return;

        this._maskCanvas.width = this._thickness.value * 2;
        this._maskCanvas.height = this._thickness.value * 2;

        let gradient = this._maskCtx.createRadialGradient(
            this._thickness.value,
            this._thickness.value,
            Math.min(this._thickness.value * this._hardness.value, this._thickness.value - 1),
            this._thickness.value,
            this._thickness.value,
            this._thickness.value,
        );
        gradient.addColorStop(0, "#ffffffff");
        gradient.addColorStop(1, "#ffffff00");

        this._maskCtx.fillStyle = gradient;
        this._maskCtx.globalAlpha = this._strength.value;
        this._maskCtx.globalCompositeOperation = "destination-in";

        this._lastPosition = this.getEventMousePosition(this._mainCanvas, event);

        this.onMouseMove(event);
    }

    // TODO: While linear interpolation is possible, larger strokes can produce
    // massive performance spikes
    onMouseMove(event) {
        let [lx, ly] = this._lastPosition;
        let [cx, cy] = this.getEventMousePosition(this._mainCanvas, event);

        this._maskCtx.putImageData(this._mainCtx.getImageData(
            Math.round(lx - this._thickness.value),
            Math.round(ly - this._thickness.value),
            this._maskCanvas.width,
            this._maskCanvas.height,
        ), 0, 0);
        this._maskCtx.fillRect(
            0,
            0,
            this._maskCanvas.width,
            this._maskCanvas.height,
        );

        this._mainCtx.drawImage(
            this._maskCanvas,
            Math.round(cx - this._thickness.value),
            Math.round(cy - this._thickness.value),
        );

        this._lastPosition = [cx, cy];
    }
}
TOOLS.smudge = SmudgeTool;

class MoveTool extends CanvasTool {
    static name = "Move";
    static icon = "\u{f0b2}";

    onMouseDown(event) {
        if (event.button != 0) return;

        this._data = this._mainCtx.getImageData(0, 0, this._mainCanvas.width, this._mainCanvas.height);
        this._mainCtx.clearRect(0, 0, this._mainCanvas.width, this._mainCanvas.height);

        this._initialPosition = this.getEventMousePosition(this._mainCanvas, event);

        this.onMouseMove(event);
    }

    onMouseMove(event) {
        let [ix, iy] = this._initialPosition;
        let [cx, cy] = this.getEventMousePosition(this._mainCanvas, event);
        let [dx, dy] = [cx - ix, cy - iy];

        let w = this._mainCanvas.width;
        let h = this._mainCanvas.height;

        let x1 = dx % w;
        let y1 = dy % h;
        let x2 = x1 - (x1 >= 0 ? w : -w);
        let y2 = y1 - (y1 >= 0 ? h : -h);

        this._overlayCtx.clearRect(0, 0, this._overlayCanvas.width, this._overlayCanvas.height);
        this._overlayCtx.putImageData(this._data, x1, y1);
        this._overlayCtx.putImageData(this._data, x2, y1);
        this._overlayCtx.putImageData(this._data, x1, y2);
        this._overlayCtx.putImageData(this._data, x2, y2);
    }

    onMouseUp(event) {
        this._mainCtx.drawImage(this._overlayCanvas, 0, 0);

        this._overlayCtx.clearRect(0, 0, this._overlayCanvas.width, this._overlayCanvas.height);
    }
}
TOOLS.move = MoveTool;

class FlipHTool extends CanvasTool {
    static name = "Flip horizontally";
    static icon = "\u{f0ec}";

    onMouseDown(event) {
        if (event.button != 0) return;

        this._overlayCtx.drawImage(this._mainCanvas, 0, 0);

        this._mainCtx.clearRect(0, 0, this._mainCanvas.width, this._mainCanvas.height);
        this._mainCtx.save();
        this._mainCtx.translate(this._mainCanvas.width, 0);
        this._mainCtx.scale(-1.0, 1.0);
        this._mainCtx.drawImage(this._overlayCanvas, 0, 0);
        this._mainCtx.restore();

        this._overlayCtx.clearRect(0, 0, this._overlayCanvas.width, this._overlayCanvas.height);
    }
}
TOOLS.flipH = FlipHTool;

class FlipVTool extends CanvasTool {
    static name = "Flip vertically";
    static icon = "\u{e099}";

    onMouseDown(event) {
        if (event.button != 0) return;

        this._overlayCtx.drawImage(this._mainCanvas, 0, 0);

        this._mainCtx.clearRect(0, 0, this._mainCanvas.width, this._mainCanvas.height);
        this._mainCtx.save();
        this._mainCtx.translate(0, this._mainCanvas.height);
        this._mainCtx.scale(1.0, -1.0);
        this._mainCtx.drawImage(this._overlayCanvas, 0, 0);
        this._mainCtx.restore();

        this._overlayCtx.clearRect(0, 0, this._overlayCanvas.width, this._overlayCanvas.height);
    }
}
TOOLS.flipV = FlipVTool;

export class CanvasWidget extends Block {
    constructor() {
        super();

        this.onValueChange = new Signal();

        this._value = null;
        this._toolInstance = null;

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

                this._mainCtx = e.getContext("2d", {willReadFrequently: true});
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

    get tool() {
        return this._toolInstance;
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

    set tool(value) {
        this._toolInstance = new TOOLS[value](this._mainCanvas, this._overlayCanvas);
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

    _onMouseDown(event) {
        if (!this._toolInstance) return;

        this._mainCtx.save();
        this._overlayCtx.save();

        this._toolInstance.onMouseDown(event);

        currentWidget = this;
    }

    _onMouseMove(event) {
        if (!this._toolInstance) return;

        this._toolInstance.onMouseMove(event);
    }

    _onMouseUp(event) {
        if (!this._toolInstance) return;

        this._toolInstance.onMouseUp(event);

        this._overlayCtx.restore();
        this._mainCtx.restore();

        this._value = this._mainCanvas.toDataURL("image/png");
        this.onValueChange.fire(this._value);

        currentWidget = null;
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
