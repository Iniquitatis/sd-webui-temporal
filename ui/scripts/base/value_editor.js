import {Block} from "../../scripts/base/block.js";
import {Row} from "../../scripts/base/row.js";
import {Widget} from "../../scripts/core/widget.js";

export class ValueEditor extends Widget {
    constructor() {
        super();

        this.style.width = "100%";

        this._header = super.createChild(Block, (e) => {
            e.style.alignItems = "center";
            e.style.display = "flex";
            e.style.justifyContent = "space-between";

            this._label = e.createChild(Block, (e) => {
                e.style.alignContent = "center";
                e.style.color = "var(--hint-color)";
                e.style.fontSize = "0.9rem";
                e.style.height = "var(--widget-height)";
            });
        });

        this._content = super.createChild(Row);
    }

    get label() {
        return this._label.innerText;
    }

    set label(value) {
        this._label.innerText = value;
    }

    createChild(tagOrClass, initializer, ...args) {
        return this._content.createChild(tagOrClass, initializer, ...args);
    }
}
