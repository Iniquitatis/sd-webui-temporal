import {ValueEditor} from "../../scripts/base/value_editor.js";
import {Signal} from "../../scripts/core/signal.js";
import {createElement} from "../../scripts/utils/dom.js";

export class NumberBox extends ValueEditor {
    constructor() {
        super();

        this._input = createElement(this._content, "input", (e) => {
            e.type = "number";
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
