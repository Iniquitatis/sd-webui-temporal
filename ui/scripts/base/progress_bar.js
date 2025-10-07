import {Block} from "/scripts/base/block.js";
import {clamp, normalize} from "/scripts/utils/math.js";

export class ProgressBar extends Block {
    constructor() {
        super();

        this._total = 1;
        this._value = 0;

        this.style.backgroundColor = "var(--input-color)";
        this.style.border = "var(--thin-border)";
        this.style.borderRadius = "var(--corners)";
        this.style.height = "var(--widget-height)";
        this.style.overflow = "hidden";
        this.style.position = "relative";

        this._fill = this.createChild(Block, (e) => {
            e.style.backgroundColor = "var(--fill-color)";
            e.style.height = "100%";
            e.style.inset = "0";
            e.style.position = "relative";
            e.style.width = "0%";
        });

        this._caption = this.createChild(Block, (e) => {
            e.style.alignContent = "center";
            e.style.inset = "0";
            e.style.padding = "0 var(--horizontal-padding)";
            e.style.position = "absolute";
            e.style.textAlign = "center";
        });
    }

    get text() {
        return this._caption.innerText;
    }

    get total() {
        return this._total;
    }

    get value() {
        return this._value;
    }

    set fillColor(value) {
        this._fill.style.backgroundColor = value;
    }

    set text(value) {
        this._caption.innerText = value;
    }

    set total(value) {
        this._total = value;
        this._updateFill();
    }

    set value(value) {
        this._value = value;
        this._updateFill();
    }

    _updateFill() {
        this._fill.style.width = `${clamp(normalize(this._value, 0.0, this._total), 0.0, 1.0) * 100.0}%`;
    }
}
customElements.define("ce-progress-bar", ProgressBar);
