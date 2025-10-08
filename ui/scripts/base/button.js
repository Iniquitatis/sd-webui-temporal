import {Signal} from "/scripts/core/signal.js";
import {Widget} from "/scripts/core/widget.js";
import {defineElement} from "/scripts/utils/dom.js";

export class Button extends Widget {
    static tag = "ce-button";
    static css = `
        <self> {
            align-content: center;
            background-color: var(--input-color);
            border: var(--thin-border);
            border-radius: var(--corners);
            font-weight: 500;
            height: var(--widget-height);
            padding: 0 var(--horizontal-padding);
            text-align: center;
            user-select: none;
        }

        <self>:hover {
            background-color: var(--hover-color);
        }
    `;

    constructor() {
        super();

        this.onClick = new Signal();
        this.addEventListener("click", () => {
            this.onClick.fire();
        });
    }

    get label() {
        return this.innerText;
    }

    set label(value) {
        this.innerText = value;
    }
}
defineElement(Button);
