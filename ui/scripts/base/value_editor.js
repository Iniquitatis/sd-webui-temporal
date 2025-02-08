import {Widget} from "../../scripts/core/widget.js";
import {createElement} from "../../scripts/utils/dom.js";

export class ValueEditor extends Widget {
    constructor() {
        super();

        this.style.width = "100%";

        this._header = createElement(this, "div", (e) => {
            e.style.alignItems = "center";
            e.style.display = "flex";
            e.style.justifyContent = "space-between";

            this._label = createElement(e, "label", (e) => {
                e.style.color = "var(--hint-color)";
                e.style.fontSize = "0.9rem";
                e.style.padding = "var(--vertical-padding) 0";
            });
        });

        this._content = createElement(this, "div", (e) => {
            e.style.display = "flex";
            e.style.flexDirection = "row";
            e.style.gap = "var(--layout-gap)";
        });
    }

    get label() {
        return this._label.innerText;
    }

    set label(value) {
        this._label.innerText = value;
    }

    createChild(cls, initializer, ...args) {
        return createElement(this._content, cls, initializer, ...args);
    }
}
