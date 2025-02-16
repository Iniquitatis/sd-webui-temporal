import {Signal} from "../../scripts/core/signal.js";
import {Widget} from "../../scripts/core/widget.js";

export class NumberBox extends Widget {
    constructor() {
        super();

        this.onValueChange = new Signal();

        this._input = this.createChild("input", (e) => {
            e.type = "number";
            e.style.width = "100%";
            e.addEventListener("input", () => {
                this.onValueChange.fire(e.valueAsNumber);
            });
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
    }

    set minimum(value) {
        this._input.min = value;
    }

    set step(value) {
        this._input.step = value;
    }

    set value(value) {
        this._input.valueAsNumber = value;

        this.onValueChange.fire(this.value);
    }
}
customElements.define("number-box", NumberBox);
