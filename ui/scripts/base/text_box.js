import {ValueEditor} from "../../scripts/base/value_editor.js";
import {Signal} from "../../scripts/core/signal.js";
import {createElement} from "../../scripts/utils/dom.js";

export class TextBox extends ValueEditor {
    constructor() {
        super();

        this._input = createElement(this._content, "input", (e) => {
            e.type = "text";
            e.style.width = "100%";
            e.addEventListener("change", () => {
                this.onValueChange.fire(e.value);
            });
        });

        this.onValueChange = new Signal();
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
