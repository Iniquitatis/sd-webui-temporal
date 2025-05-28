import {Block} from "../../scripts/base/block.js";
import {DragController} from "../../scripts/core/drag_controller.js";
import {Signal} from "../../scripts/core/signal.js";
import {clamp, countFractionDigits, lerp, normalize, quantize} from "../../scripts/utils/math.js";

let drag = new DragController();
drag.onMove.connect((element, event) => {
    let rect = element.getBoundingClientRect();
    element._quantizedValue = lerp(
        element._minimum,
        element._maximum,
        clamp(event.clientX - rect.left, 0, rect.width) / rect.width,
    );
    element._updateElements();
});
drag.onEnd.connect((element) => {
    element.onValueChange.fire(element._value);
});

export class Slider extends Block {
    constructor() {
        super();

        this.onValueChange = new Signal();

        this._minimum = 0;
        this._maximum = 1;
        this._step = 1;
        this._value = 0;

        let activateInput = () => {
            drag.enabled = false;
            this._caption.visible = false;
            this._input.max = this._maximum;
            this._input.min = this._minimum;
            this._input.step = this._step;
            this._input.value = this._fixedValueString;
            this._input.style.display = "block";
            this._input.focus();
        };

        let deactivateInput = () => {
            this._input.style.display = "none";
            this._caption.visible = true;
            drag.enabled = true;
            this.onValueChange.fire(this._value);
        };

        this.tabIndex = 0;
        this.style.backgroundColor = "var(--input-color)";
        this.style.border = "var(--thin-border)";
        this.style.borderRadius = "var(--corners)";
        this.style.cursor = "pointer";
        this.style.height = "var(--widget-height)";
        this.style.overflow = "hidden";
        this.style.position = "relative";
        this.style.userSelect = "none";
        this.addEventListener("dblclick", activateInput);
        this.addEventListener("keydown", (event) => {
            switch (event.key) {
                case "ArrowLeft": {
                    this._quantizedValue = clamp(this._value - this._step, this._minimum, this._maximum);
                    this._updateElements();
                    this.onValueChange.fire(this._value);
                } break;

                case "ArrowRight": {
                    this._quantizedValue = clamp(this._value + this._step, this._minimum, this._maximum);
                    this._updateElements();
                    this.onValueChange.fire(this._value);
                } break;

                case "Enter": {
                    activateInput();
                } break;
            }
        });
        drag.register(this);

        this._fill = this.createChild(Block, (e) => {
            e.style.backgroundColor = "var(--fill-color)";
            e.style.height = "100%";
            e.style.inset = "0";
            e.style.pointerEvents = "none";
            e.style.position = "relative";
            e.style.width = "0%";
        });

        this._caption = this.createChild(Block, (e) => {
            e.style.alignContent = "center";
            e.style.inset = "0";
            e.style.padding = "0 var(--horizontal-padding)";
            e.style.pointerEvents = "none";
            e.style.position = "absolute";
        });

        this._input = this.createChild("input", (e) => {
            e.type = "number";
            e.style.alignContent = "center";
            e.style.background = "none";
            e.style.border = "none";
            e.style.display = "none";
            e.style.height = "100%";
            e.style.inset = "0";
            e.style.padding = "0 var(--horizontal-padding)";
            e.style.position = "absolute";
            e.style.textAlign = "left";
            e.addEventListener("input", () => {
                this._quantizedValue = e.valueAsNumber;
                this._updateElements();
            });
            e.addEventListener("keydown", (event) => {
                if (event.key == "Enter") event.stopPropagation();
                if (["Enter", "Escape"].includes(event.key)) deactivateInput();
            });
            e.addEventListener("blur", () => {
                deactivateInput();
            });
        });
    }

    get maximum() {
        return this._maximum;
    }

    get minimum() {
        return this._minimum;
    }

    get step() {
        return this._step;
    }

    get value() {
        return this._value;
    }

    set fillColor(value) {
        this._fill.style.backgroundColor = value;
    }

    set maximum(value) {
        this._maximum = value;
        this._updateElements();
    }

    set minimum(value) {
        this._minimum = value;
        this._updateElements();
    }

    set step(value) {
        this._step = value;
        this._updateElements();
    }

    set value(value) {
        this._quantizedValue = value;
        this._updateElements();
        this.onValueChange.fire(value);
    }

    get _fixedValueString() {
        return this._value.toFixed(countFractionDigits(this._step));
    }

    set _quantizedValue(value) {
        this._value = lerp(
            this._minimum,
            this._maximum,
            quantize(
                normalize(value, this._minimum, this._maximum),
                this._step / (this._maximum - this._minimum),
            ),
        );
    }

    _updateElements() {
        this._fill.style.width = `${clamp(normalize(this._value, this._minimum, this._maximum), 0.0, 1.0) * 100.0}%`;
        this._caption.innerText = this._fixedValueString;
    }
}
customElements.define("custom-slider", Slider);
