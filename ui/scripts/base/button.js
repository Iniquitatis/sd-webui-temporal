import {Signal} from "../../scripts/core/signal.js";
import {Widget} from "../../scripts/core/widget.js";
import {createElement} from "../../scripts/utils/dom.js";

export class Button extends Widget {
    constructor() {
        super();

        this.style.width = "100%";

        this._button = createElement(this, "button", (e) => {
            e.style.height = "100%";
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
