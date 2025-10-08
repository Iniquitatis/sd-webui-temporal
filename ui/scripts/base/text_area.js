import {Signal} from "/scripts/core/signal.js";
import {Widget} from "/scripts/core/widget.js";
import {defineElement} from "/scripts/utils/dom.js";

export class TextArea extends Widget {
    static tag = "ce-text-area";
    static css = `
        <self> > textarea {
            display: block;
            resize: vertical;
            width: 100%;
        }
    `;

    constructor() {
        super();

        this.onValueChange = new Signal();

        this._textArea = this.createChild("textarea", (e) => {
            e.rows = 5;
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
defineElement(TextArea);
