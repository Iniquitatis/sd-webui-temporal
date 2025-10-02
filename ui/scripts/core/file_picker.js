import {Signal} from "/scripts/core/signal.js";
import {createElement} from "/scripts/utils/dom.js";

export class FilePicker {
    constructor() {
        this.onLoad = new Signal();

        this._input = createElement(null, "input", (e) => {
            e.type = "file";
            e.addEventListener("change", () => {
                let reader = new FileReader();
                reader.addEventListener("load", () => {
                    this.onLoad.fire(reader.result);

                    e.value = null;
                });
                reader.readAsDataURL(e.files[0]);
            });
        });
    }

    get mimeType() {
        return this._input.accept;
    }

    set mimeType(value) {
        this._input.accept = value;
    }

    open() {
        this._input.click();
    }
}
