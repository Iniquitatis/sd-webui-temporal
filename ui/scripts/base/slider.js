import {ValueEditor} from "../../scripts/base/value_editor.js";
import {Signal} from "../../scripts/core/signal.js";
import {createElement} from "../../scripts/utils/dom.js";

export class Slider extends ValueEditor {
    constructor() {
        super();

        this._headerInput = createElement(this._header, "input", (e) => {
            e.type = "number";
            e.style.textAlign = "right";
            e.style.width = "var(--small-input-width)";
            e.addEventListener("input", () => {
                this._input.valueAsNumber = e.valueAsNumber;

                this.onValueChange.fire(e.valueAsNumber);
            });
        });

        this._input = createElement(this._content, "input", (e) => {
            e.type = "range";
            e.style.width = "100%";
            e.addEventListener("input", () => {
                this._headerInput.valueAsNumber = e.valueAsNumber;

                this.onValueChange.fire(e.valueAsNumber);
            });
        });

        this.onValueChange = new Signal();
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
        this._headerInput.max = value;
    }

    set minimum(value) {
        this._input.min = value;
        this._headerInput.min = value;
    }

    set step(value) {
        this._input.step = value;
        this._headerInput.step = value;
    }

    set value(value) {
        this._input.valueAsNumber = value;
        this._headerInput.valueAsNumber = value;

        this.onValueChange.fire(this.value);
    }
}
customElements.define("custom-slider", Slider);
