import {Button} from "../../scripts/base/button.js";

export class ToolButton extends Button {
    constructor() {
        super();

        this.style.height = "var(--widget-height)";
        this.style.width = "var(--widget-height)";

        this._button.style.padding = "unset";
    }
}
customElements.define("tool-button", ToolButton);
