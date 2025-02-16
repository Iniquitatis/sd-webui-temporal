import {Signal} from "../../scripts/core/signal.js";
import {Widget} from "../../scripts/core/widget.js";

export class TextBox extends Widget {
    constructor() {
        super();

        this.onValueChange = new Signal();

        this._input = this.createChild("input", (e) => {
            e.type = "text";
            e.style.width = "100%";
            e.addEventListener("change", () => {
                this.onValueChange.fire(e.value);
            });
        });
    }

    get value() {
        return this._input.value;
    }

    set value(value) {
        this._input.value = value;

        this.onValueChange.fire(this.value);
    }
}
customElements.define("text-box", TextBox);
