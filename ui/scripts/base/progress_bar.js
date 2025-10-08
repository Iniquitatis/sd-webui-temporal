import {Block} from "/scripts/base/block.js";
import {defineElement} from "/scripts/utils/dom.js";
import {clamp, normalize} from "/scripts/utils/math.js";

export class ProgressBar extends Block {
    static tag = "ce-progress-bar";
    static css = `
        <self> {
            background-color: var(--input-color);
            border: var(--thin-border);
            border-radius: var(--corners);
            height: var(--widget-height);
            overflow: hidden;
            position: relative;
        }

        <self> > .fill {
            background-color: var(--fill-color);
            height: 100%;
            inset: 0;
            position: relative;
        }

        <self> > .caption {
            align-content: center;
            inset: 0;
            padding: 0 var(--horizontal-padding);
            position: absolute;
            text-align: center;
        }
    `;

    constructor() {
        super();

        this._total = 1;
        this._value = 0;

        this._fill = this.createChild(Block, (e) => {
            e.className = "fill";
            e.style.width = "0%";
        });

        this._caption = this.createChild(Block, (e) => {
            e.className = "caption";
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
defineElement(ProgressBar);
