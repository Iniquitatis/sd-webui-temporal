import {ValueEditor} from "../../scripts/base/value_editor.js";
import {Signal} from "../../scripts/core/signal.js";

export class Checkbox extends ValueEditor {
    constructor() {
        super();

        this.onValueChange = new Signal();

        this.addEventListener("click", () => {
            this._input.checked = !this._input.checked;

            this.onValueChange.fire(this._input.checked);
        });

        this._input = this._header.createChild("input", (e) => {
            e.type = "checkbox";
            e.style.marginLeft = "var(--horizontal-padding)";
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

    createChild(tagOrClass, initializer, ...args) {
        return this._header.createChild(tagOrClass, initializer, ...args);
    }
}
customElements.define("custom-checkbox", Checkbox);
