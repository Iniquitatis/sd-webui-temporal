import {ValueEditor} from "../../scripts/base/value_editor.js";
import {Signal} from "../../scripts/core/signal.js";
import {createElement} from "../../scripts/utils/dom.js";

export class ImageEditor extends ValueEditor {
    constructor() {
        super();

        this._content.style.height = "calc(100% - var(--widget-height))";

        createElement(this._content, "div", (e) => {
            e.style.alignContent = "center";
            e.style.border = "var(--thin-border)";
            e.style.borderRadius = "var(--corners)";
            e.style.minHeight = "10rem";
            e.style.textAlign = "center";
            e.style.userSelect = "none";
            e.style.width = "100%";

            this._input = createElement(e, "input", (e) => {
                e.type = "file";
                e.style.height = "100%";
                e.style.width = "100%";
                e.addEventListener("change", () => {
                    this.onValueChange.fire(e.value);
                });
            });

            this._img = createElement(e, "img", (e) => {
                e.style.display = "none";
                e.style.height = "100%";
                e.style.maxWidth = "100%";
                e.style.objectFit = "contain";
            });
        });

        this.onValueChange = new Signal();
    }

    get value() {
        return this._img.src;
    }

    set height(value) {
        this.style.height = value;
    }

    set value(value) {
        this._input.style.display = value ? "none" : null;

        this._img.src = value ?? null;
        this._img.style.display = value ? null : "none";

        this.onValueChange.fire(this.value);
    }
}
customElements.define("image-editor", ImageEditor);
