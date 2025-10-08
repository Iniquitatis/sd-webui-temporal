import {Block} from "/scripts/base/block.js";
import {DragController} from "/scripts/core/drag_controller.js";
import {Signal} from "/scripts/core/signal.js";
import {defineElement} from "/scripts/utils/dom.js";
import {clamp, countFractionDigits, lerp, normalize, quantize} from "/scripts/utils/math.js";

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
    static tag = "ce-slider";
    static css = `
        <self> {
            background-color: var(--input-color);
            border: var(--thin-border);
            border-radius: var(--corners);
            cursor: pointer;
            height: var(--widget-height);
            overflow: hidden;
            position: relative;
            user-select: none;
        }

        <self> > .fill {
            background-color: var(--fill-color);
            height: 100%;
            inset: 0;
            pointer-events: none;
            position: relative;
        }

        <self> > .caption {
            align-content: center;
            inset: 0;
            padding: 0 var(--horizontal-padding);
            pointer-events: none;
            position: absolute;
        }

        <self> > input {
            align-content: center;
            background: none;
            border: none;
            height: 100%;
            inset: 0;
            padding: 0 var(--horizontal-padding);
            position: absolute;
            text-align: left;
        }
    `;

    constructor() {
        super();

        this.onValueChange = new Signal();

        this._minimum = 0;
        this._maximum = 1;
        this._step = 1;
        this._suffix = "";
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
            e.className = "fill";
            e.style.width = "0%";
        });

        this._caption = this.createChild(Block, (e) => {
            e.className = "caption";
        });

        this._input = this.createChild("input", (e) => {
            e.type = "number";
            e.style.display = "none";
            e.addEventListener("input", () => {
                this._quantizedValue = e.valueAsNumber;
                this._updateElements();
            });
            e.addEventListener("keydown", (event) => {
                switch (event.key) {
                    case "Enter": {
                        event.preventDefault();
                        event.stopPropagation();
                        e.blur();
                    } break;

                    case "Escape": {
                        e.blur();
                    } break;
                }
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

    get suffix() {
        return this._suffix;
    }

    get value() {
        return this._value;
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

    set suffix(value) {
        this._suffix = value;
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
        this._value = parseFloat(lerp(
            this._minimum,
            this._maximum,
            quantize(
                normalize(value, this._minimum, this._maximum),
                this._step / (this._maximum - this._minimum),
            ),
        ).toFixed(countFractionDigits(this._step)));
    }

    _updateElements() {
        this._fill.style.width = `${clamp(normalize(this._value, this._minimum, this._maximum), 0.0, 1.0) * 100.0}%`;
        this._caption.innerText = `${this._fixedValueString}${this._suffix}`;
    }
}
defineElement(Slider);
