import {ValueEditor} from "../../scripts/base/value_editor.js";
import {Signal} from "../../scripts/core/signal.js";
import {createElement} from "../../scripts/utils/dom.js";

export class CodeArea extends ValueEditor {
    constructor() {
        super();

        this._textArea = createElement(this._content, "textarea", (e) => {
            e.rows = 5;
            e.style.display = "block";
            e.style.fontFamily = "monospace";
            e.style.width = "100%";
            e.addEventListener("change", () => {
                this.onValueChange.fire(e.value);
            });
            e.addEventListener("keydown", (event) => {
                if (event.key != "Tab") return;

                event.preventDefault();

                let start = e.selectionStart;
                let end = e.selectionEnd;
                let indentation = "    ";

                e.value = `${e.value.substring(0, start)}${indentation}${e.value.substring(end)}`;

                e.selectionStart = start + indentation.length;
                e.selectionEnd = start + indentation.length;

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
customElements.define("code-area", CodeArea);
