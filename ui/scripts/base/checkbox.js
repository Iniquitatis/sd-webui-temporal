import {Block} from "/scripts/base/block.js";
import {Signal} from "/scripts/core/signal.js";
import {Widget} from "/scripts/core/widget.js";
import {defineElement} from "/scripts/utils/dom.js";

export class Checkbox extends Widget {
    static tag = "ce-checkbox";
    static css = `
        <self> {
            display: block;
            height: var(--widget-height);
        }

        <self>.with-title {
            display: flex;
            flex-direction: row;
            gap: var(--layout-small-gap);
        }

        <self>.with-title > ce-block {
            align-content: center;
            color: var(--hint-color);
            font-size: var(--hint-size);
            height: var(--widget-height);
            user-select: none;
        }
    `;

    constructor() {
        super();

        this.onValueChange = new Signal();

        this.addEventListener("click", () => {
            this._input.checked = !this._input.checked;

            this.onValueChange.fire(this._input.checked);
        });

        this._input = this.createChild("input", (e) => {
            e.type = "checkbox";
            e.addEventListener("click", (event) => {
                event.stopPropagation();

                this.onValueChange.fire(e.checked);
            });
        });
    }

    get value() {
        return this._input.checked;
    }

    set value(value) {
        this._input.checked = value;

        this.onValueChange.fire(this.value);
    }

    attachTitle(title) {
        this.classList.add("with-title");

        this.createChild(Block, (e) => {
            e.innerText = title;
        });
    }

    canAttachTitle() {
        return true;
    }
}
defineElement(Checkbox);
