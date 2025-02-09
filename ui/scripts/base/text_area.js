import {ValueEditor} from "../../scripts/base/value_editor.js";
import {Signal} from "../../scripts/core/signal.js";
import {createElement} from "../../scripts/utils/dom.js";

export class TextArea extends ValueEditor {
    constructor() {
        super();

        this._textArea = createElement(this._content, "textarea", (e) => {
            e.rows = 5;
            e.style.display = "block";
            e.style.width = "100%";
            e.addEventListener("change", () => {
                this.onValueChange.fire(e.value);
            });
        });

        this.onValueChange = new Signal();
    }

    get value() {
        return this._textArea.value;
    }

    set value(value) {
        this._textArea.value = value;

        this.onValueChange.fire(this.value);
    }
}
customElements.define("text-area", TextArea);
