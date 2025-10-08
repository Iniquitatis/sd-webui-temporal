import {Block} from "/scripts/base/block.js";
import {Row} from "/scripts/base/row.js";
import {Signal} from "/scripts/core/signal.js";
import {Widget} from "/scripts/core/widget.js";
import {defineElement} from "/scripts/utils/dom.js";

export class Overlay extends Widget {
    static tag = "ce-overlay";
    static css = `
        <self> {
            align-content: center;
            background: hsla(0 0% 0% / 75%);
            inset: 0;
            position: fixed;
            z-index: 1;
        }

        <self> > ce-block {
            inset: 0;
            position: absolute;
        }

        <self> > ce-row {
            flex-direction: row-reverse;
            position: absolute;
            right: 0;
            top: 0;
            width: auto;
        }

        <self> > ce-row > ce-block {
            align-content: center;
            color: white;
            cursor: pointer;
            font-size: 3rem;
            font-weight: bold;
            height: 3rem;
            text-align: center;
            width: 3rem;
        }
    `;

    constructor() {
        super();

        this.onClose = new Signal();

        this._content = super.createChild(Block);

        this._tools = super.createChild(Row);

        this.addTool("\u{f00d}", () => this.close());
    }

    addTool(icon, callback) {
        this._tools.createChild(Block, (e) => {
            e.classList.add("transhover");
            e.innerText = icon;
            e.addEventListener("click", callback);
        });
    }

    close() {
        this.onClose.fire();

        this.parentElement.removeChild(this);
    }

    createChild(tagOrClass, initializer, ...args) {
        return this._content.createChild(tagOrClass, initializer, ...args);
    }
}
defineElement(Overlay);
