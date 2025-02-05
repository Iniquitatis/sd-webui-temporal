import {Signal} from "../../scripts/core/signal.js";
import {Widget} from "../../scripts/core/widget.js";
import {createElement} from "../../scripts/utils/dom.js";

export class Button extends Widget {
    constructor() {
        super();

        this.style.width = "100%";

        this._button = createElement(this, "button", (e) => {
            e.style.width = "100%";
            e.addEventListener("click", () => {
                this.onClick.fire();
            });
        });

        this.onClick = new Signal();
    }

    get label() {
        return this._button.innerText;
    }

    set label(value) {
        this._button.innerText = value;
    }
}
customElements.define("custom-button", Button);

export class ToolButton extends Button {
    constructor() {
        super();

        this.style.height = "var(--widget-height)";
        this.style.width = "var(--widget-height)";

        this._button.style.height = "100%";
        this._button.style.padding = "0";
        this._button.style.width = "100%";
    }
}
customElements.define("tool-button", ToolButton);

export class MultiStateToggle extends Widget {
    constructor(states) {
        super();

        // NOTE: Yes, `states` is a `value:caption` dictionary
        this._values = Object.keys(states);
        this._captions = states;
        this._index = 0;

        this.style.alignContent = "center";
        this.style.cursor = "pointer";
        this.style.height = "var(--widget-height)";
        this.style.textAlign = "center";
        this.style.userSelect = "none";
        this.style.width = "var(--widget-height)";
        this.addEventListener("click", () => {
            this._index++;
            this._index %= this._values.length;
            this._refresh();

            this.onValueChange.fire(this.value);
        });

        this.onValueChange = new Signal();

        this._refresh();
    }

    get value() {
        return this._values[this._index];
    }

    set value(value) {
        this._index = this._values.indexOf(value);
        this._refresh();

        this.onValueChange.fire(this.value);
    }

    _refresh() {
        this.innerText = this._captions[this.value];
    }
}
customElements.define("multi-state-toggle", MultiStateToggle);
