import {Signal} from "/scripts/core/signal.js";
import {Widget} from "/scripts/core/widget.js";

export class TextArea extends Widget {
    constructor() {
        super();

        this.onValueChange = new Signal();

        this._textArea = this.createChild("textarea", (e) => {
            e.rows = 5;
            e.style.display = "block";
            e.style.resize = "vertical";
            e.style.width = "100%";
            e.addEventListener("change", () => {
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
customElements.define("ce-text-area", TextArea);
