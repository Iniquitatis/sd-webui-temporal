import {Block} from "../../scripts/base/block.js";
import {Widget} from "../../scripts/core/widget.js";

export class ProgressBar extends Widget {
    constructor() {
        super();

        this.style.height = "calc(var(--widget-height) * 1.5)";
        this.style.marginTop = "calc(var(--layout-gap) * -1)";
        this.style.position = "relative";

        this._progress = this.createChild("progress", (e) => {
            e.value = 0.0;
            e.style.height = "100%";
            e.style.width = "100%";
        });

        this._text = this.createChild(Block, (e) => {
            e.style.alignContent = "center";
            e.style.color = "var(--background-color)";
            e.style.fontWeight = "bold";
            e.style.height = "100%";
            e.style.left = "0";
            e.style.position = "absolute";
            e.style.textAlign = "center";
            e.style.top = "0";
            e.style.width = "100%";
        });
    }

    get text() {
        return this._text.innerText;
    }

    get total() {
        return this._progress.max;
    }

    get value() {
        return this._progress.value;
    }

    set text(value) {
        this._text.innerText = value;
    }

    set total(value) {
        this._progress.max = value;
    }

    set value(value) {
        this._progress.value = value;
    }
}
customElements.define("progress-bar", ProgressBar);
