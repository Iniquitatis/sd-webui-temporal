import {Signal} from "../../scripts/core/signal.js";
import {Widget} from "../../scripts/core/widget.js";

export class Button extends Widget {
    constructor() {
        super();

        this.onClick = new Signal();

        this.style.height = "var(--widget-height)";
        this.style.width = "100%";

        this._button = this.createChild("button", (e) => {
            e.style.height = "100%";
            e.style.width = "100%";
            e.addEventListener("click", () => {
                this.onClick.fire();
            });
        });
    }

    get label() {
        return this._button.innerText;
    }

    set label(value) {
        this._button.innerText = value;
    }
}
customElements.define("custom-button", Button);
