import {Block} from "../../scripts/base/block.js";
import {Signal} from "../../scripts/core/signal.js";

export class VideoWidget extends Block {
    constructor() {
        super();

        this.onValueChange = new Signal();

        this.style.alignContent = "center";
        this.style.position = "relative";
        this.style.textAlign = "center";
        this.style.userSelect = "none";

        this._video = this.createChild("video", (e) => {
            e.controls = "controls";
            e.style.display = "none";
            e.style.height = "100%";
            e.style.maxWidth = "100%";
            e.style.objectFit = "contain";
            e.style.verticalAlign = "middle";
            e.style.width = "auto";
        });
    }

    get value() {
        return this._video.src || null;
    }

    set value(value) {
        // FIXME: Should receive the correct value in the first place
        if (value && !value.startsWith("data:video/")) {
            value = `data:video/mp4;base64,${value}`;
        }

        if (value) {
            this._video.src = value;
            this._video.style.display = null;
        } else {
            this._video.removeAttribute("src");
            this._video.style.display = "none";
        }

        this.onValueChange.fire(this.value);
    }
}
customElements.define("video-widget", VideoWidget);
