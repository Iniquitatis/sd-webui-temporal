import {Block} from "/scripts/base/block.js";
import {Signal} from "/scripts/core/signal.js";
import {defineElement} from "/scripts/utils/dom.js";

export class ImageWidget extends Block {
    static tag = "ce-image-widget";
    static css = `
        <self> {
            align-content: center;
            position: relative;
            text-align: center;
            user-select: none;
        }

        <self> > img {
            height: 100%;
            max-width: 100%;
            object-fit: contain;
            vertical-align: middle;
            width: auto;
        }
    `;

    constructor() {
        super();

        this.onValueChange = new Signal();

        this._img = this.createChild("img", (e) => {
            e.style.display = "none";
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
defineElement(ImageWidget);
