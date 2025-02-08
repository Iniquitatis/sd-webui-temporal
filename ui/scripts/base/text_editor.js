import {ValueEditor} from "../../scripts/base/value_editor.js";
import {Signal} from "../../scripts/core/signal.js";
import {createElement} from "../../scripts/utils/dom.js";

export class TextEditor extends ValueEditor {
    constructor() {
        super();

        this._input = createElement(this._content, "input", (e) => {
            e.type = "text";
            e.style.width = "100%";
            e.addEventListener("change", () => {
                this.onValueChange.fire(e.value);
            });
        });

        // NOTE: Intentionally disconnected
        this._textArea = createElement(null, "textarea", (e) => {
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
        return this._content.contains(this._textArea) ? this._textArea.value : this._input.value;
    }

    get variant() {
        return this._content.contains(this._textArea) ? "area" : "box";
    }

    set value(value) {
        if (this._content.contains(this._textArea)) {
            this._textArea.value = value;
        }
        else {
            this._input.value = value;
        }

        this.onValueChange.fire(this.value);
    }

    set variant(value) {
        if (value == "area" && this._content.contains(this._input) && !this._content.contains(this._textArea)) {
            this._content.removeChild(this._input);
            this._content.appendChild(this._textArea);
        } else if (this._content.contains(this._textArea) && !this._content.contains(this._input)) {
            this._content.removeChild(this._textArea);
            this._content.appendChild(this._input);
        }
    }
}
customElements.define("text-editor", TextEditor);
