import {Signal} from "../../scripts/core/signal.js";
import {Widget} from "../../scripts/core/widget.js";

export class CodeArea extends Widget {
    constructor() {
        super();

        this.onValueChange = new Signal();

        this._textArea = this.createChild("textarea", (e) => {
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
