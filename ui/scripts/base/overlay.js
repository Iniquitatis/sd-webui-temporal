import {Block} from "../../scripts/base/block.js";
import {Row} from "../../scripts/base/row.js";
import {Signal} from "../../scripts/core/signal.js";
import {Widget} from "../../scripts/core/widget.js";

export class Overlay extends Widget {
    constructor() {
        super();

        this.onClose = new Signal();

        this.style.alignContent = "center";
        this.style.background = "hsla(0 0% 0% / 75%)";
        this.style.inset = "0";
        this.style.position = "fixed";
        this.style.zIndex = "1";

        this._content = super.createChild(Block, (e) => {
            e.style.inset = "0";
            e.style.position = "absolute";
        });

        this._tools = super.createChild(Row, (e) => {
            e.style.flexDirection = "row-reverse";
            e.style.position = "absolute";
            e.style.right = "0";
            e.style.top = "0";
            e.style.width = "auto";
        });

        this.addTool("\u{f00d}", () => this.close());
    }

    addTool(icon, callback) {
        this._tools.createChild(Block, (e) => {
            e.classList.add("transhover");
            e.innerText = icon;
            e.style.alignContent = "center";
            e.style.color = "white";
            e.style.cursor = "pointer";
            e.style.fontSize = "3rem";
            e.style.fontWeight = "bold";
            e.style.height = "3rem";
            e.style.textAlign = "center";
            e.style.width = "3rem";
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
customElements.define("overlay-widget", Overlay);
