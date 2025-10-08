import {Block} from "/scripts/base/block.js";
import {createElement, defineElement} from "/scripts/utils/dom.js";

export class GroupBox extends Block {
    static tag = "ce-group-box";
    static css = `
        <self> > fieldset {
            border: var(--thin-border);
            border-radius: var(--corners);
            padding: var(--layout-padding);
        }

        <self> > fieldset > legend {
            color: var(--hint-color);
            font-size: 0.9rem;
        }
    `;

    constructor() {
        super();

        this._fieldset = super.createChild("fieldset", (e) => {
            this._legend = createElement(e, "legend");
        });
    }

    get label() {
        return this._legend.innerText;
    }

    set label(value) {
        this._legend.innerText = value;
    }

    createChild(tagOrClass, initializer, ...args) {
        return createElement(this._fieldset, tagOrClass, initializer, ...args);
    }
}
defineElement(GroupBox);
