import {Signal} from "/scripts/core/signal.js";
import {Widget} from "/scripts/core/widget.js";
import {defineElement} from "/scripts/utils/dom.js";

export class TextBox extends Widget {
    static tag = "ce-text-box";
    static css = `
        <self> > input {
            width: 100%;
        }
    `;

    constructor() {
        super();

        this.onValueChange = new Signal();

        this._input = this.createChild("input", (e) => {
            e.type = "text";
            e.addEventListener("change", () => {
                this.onValueChange.fire(e.value);
            });
        });
    }

    get value() {
        return this._input.value;
    }

    set value(value) {
        this._input.value = value;

        this.onValueChange.fire(this.value);
    }
}
defineElement(TextBox);
