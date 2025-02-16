import {Signal} from "../../scripts/core/signal.js";
import {Widget} from "../../scripts/core/widget.js";

export class Checkbox extends Widget {
    constructor() {
        super();

        this.onValueChange = new Signal();

        this.style.display = "block";
        this.style.height = "var(--widget-height)";
        this.addEventListener("click", () => {
            this._input.checked = !this._input.checked;

            this.onValueChange.fire(this._input.checked);
        });

        this._input = this.createChild("input", (e) => {
            e.type = "checkbox";
            e.addEventListener("click", (event) => {
                event.stopPropagation();

                this.onValueChange.fire(e.checked);
            });
        });
    }

    get value() {
        return this._input.checked;
    }

    set value(value) {
        this._input.checked = value;

        this.onValueChange.fire(this.value);
    }
}
customElements.define("custom-checkbox", Checkbox);
