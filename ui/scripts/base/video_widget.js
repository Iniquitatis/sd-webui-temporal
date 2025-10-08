import {Block} from "/scripts/base/block.js";
import {Signal} from "/scripts/core/signal.js";
import {defineElement} from "/scripts/utils/dom.js";

export class VideoWidget extends Block {
    static tag = "ce-video-widget";
    static css = `
        <self> {
            align-content: center;
            position: relative;
            text-align: center;
            user-select: none;
        }

        <self> > video {
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

        this._video = this.createChild("video", (e) => {
            e.controls = "controls";
            e.style.display = "none";
        });
    }

    get value() {
        return this._video.src || null;
    }

    set value(value) {
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
defineElement(VideoWidget);
