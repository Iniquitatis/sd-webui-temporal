import {Row} from "../../scripts/base/row.js";
import {Signal} from "../../scripts/core/signal.js";
import {Widget} from "../../scripts/core/widget.js";

export class Slider extends Widget {
    constructor(withNumber = true) {
        super();

        this.onValueChange = new Signal();

        this.createChild(Row, (e) => {
            this._input = e.createChild("input", (e) => {
                e.type = "range";
                e.style.width = "100%";
                e.addEventListener("input", () => {
                    if (this._numberInput) {
                        this._numberInput.valueAsNumber = e.valueAsNumber;
                    }
                });
                e.addEventListener("change", () => {
                    this.onValueChange.fire(e.valueAsNumber);
                });
            });

            if (withNumber) {
                this._numberInput = e.createChild("input", (e) => {
                    e.type = "number";
                    e.style.textAlign = "right";
                    e.style.width = "var(--small-input-width)";
                    e.addEventListener("input", () => {
                        this._input.valueAsNumber = e.valueAsNumber;
                    });
                    e.addEventListener("change", () => {
                        this.onValueChange.fire(e.valueAsNumber);
                    });
                });
            }
        });
    }

    get maximum() {
        return this._input.max;
    }

    get minimum() {
        return this._input.min;
    }

    get step() {
        return this._input.step;
    }

    get value() {
        return this._input.valueAsNumber;
    }

    set maximum(value) {
        this._input.max = value;

        if (this._numberInput) {
            this._numberInput.max = value;
        }
    }

    set minimum(value) {
        this._input.min = value;

        if (this._numberInput) {
            this._numberInput.min = value;
        }
    }

    set step(value) {
        this._input.step = value;

        if (this._numberInput) {
            this._numberInput.step = value;
        }
    }

    set value(value) {
        this._input.valueAsNumber = value;

        if (this._numberInput) {
            this._numberInput.valueAsNumber = value;
        }

        this.onValueChange.fire(value);
    }
}
customElements.define("custom-slider", Slider);
