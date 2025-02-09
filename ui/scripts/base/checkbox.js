import {ValueEditor} from "../../scripts/base/value_editor.js";
import {Signal} from "../../scripts/core/signal.js";
import {createElement} from "../../scripts/utils/dom.js";

export class Checkbox extends ValueEditor {
    constructor() {
        super();

        this._input = createElement(this._header, "input", (e) => {
            e.type = "checkbox";
            e.style.marginLeft = "var(--horizontal-padding)";
            e.addEventListener("click", (event) => {
                event.stopPropagation();

                this.onValueChange.fire(e.checked);
            });
        });

        this.addEventListener("click", () => {
            this._input.checked = !this._input.checked;

            this.onValueChange.fire(this._input.checked);
        });

        this.onValueChange = new Signal();
    }

    get value() {
        return this._input.checked;
    }

    set value(value) {
        this._input.checked = value;

        this.onValueChange.fire(this.value);
    }

    createChild(cls, initializer, ...args) {
        return createElement(this._header, cls, initializer, ...args);
    }
}
customElements.define("custom-checkbox", Checkbox);
