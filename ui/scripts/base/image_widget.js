import {Block} from "../../scripts/base/block.js";
import {Signal} from "../../scripts/core/signal.js";

export class ImageWidget extends Block {
    constructor() {
        super();

        this.onValueChange = new Signal();

        this.style.alignContent = "center";
        this.style.position = "relative";
        this.style.textAlign = "center";
        this.style.userSelect = "none";

        this._img = this.createChild("img", (e) => {
            e.style.display = "none";
            e.style.height = "100%";
            e.style.maxWidth = "100%";
            e.style.objectFit = "contain";
            e.style.verticalAlign = "middle";
            e.style.width = "auto";
        });
    }

    get value() {
        return this._img.src || null;
    }

    set value(value) {
        if (value) {
            this._img.src = value;
            this._img.style.display = null;
        } else {
            this._img.removeAttribute("src");
            this._img.style.display = "none";
        }

        this.onValueChange.fire(this.value);
    }
}
customElements.define("image-widget", ImageWidget);
