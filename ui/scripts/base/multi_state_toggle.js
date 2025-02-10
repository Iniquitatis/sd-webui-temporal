import {Signal} from "../../scripts/core/signal.js";
import {Widget} from "../../scripts/core/widget.js";

export class MultiStateToggle extends Widget {
    constructor(states) {
        super();

        this._values = Object.keys(states);
        this._labels = Object.values(states);
        this._index = 0;

        this.innerText = this._labels[0];
        this.style.alignContent = "center";
        this.style.cursor = "pointer";
        this.style.height = "var(--widget-height)";
        this.style.textAlign = "center";
        this.style.userSelect = "none";
        this.style.width = "var(--widget-height)";
        this.addEventListener("click", () => {
            this._index++;
            this._index %= this._values.length;
            this.innerText = this._labels[this._index];

            this.onValueChange.fire(this.value);
        });

        this.onValueChange = new Signal();
    }

    get value() {
        return this._values[this._index];
    }

    set value(value) {
        this._index = this._values.indexOf(value);
        this.innerText = this._labels[this._index];

        this.onValueChange.fire(this.value);
    }
}
customElements.define("multi-state-toggle", MultiStateToggle);
