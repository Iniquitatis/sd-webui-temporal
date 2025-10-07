import {Block} from "/scripts/base/block.js";
import {createElement} from "/scripts/utils/dom.js";

export class GroupBox extends Block {
    constructor() {
        super();

        this._fieldset = super.createChild("fieldset", (e) => {
            e.style.border = "var(--thin-border)";
            e.style.borderRadius = "var(--corners)";
            e.style.padding = "var(--layout-padding)";

            this._legend = createElement(e, "legend", (e) => {
                e.style.color = "var(--hint-color)";
                e.style.fontSize = "0.9rem";
            });
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
customElements.define("ce-group-box", GroupBox);
