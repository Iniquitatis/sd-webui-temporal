import {Block} from "../../scripts/base/block.js";
import {ToolButton} from "../../scripts/base/tool_button.js";
import {Signal} from "../../scripts/core/signal.js";
import {Widget} from "../../scripts/core/widget.js";
import {createElement} from "../../scripts/utils/dom.js";

// FIXME: Deduplicate with ImageViewer
class VideoViewer extends Widget {
    constructor() {
        super();

        this.viewedElement = null;

        this.style.alignContent = "center";
        this.style.background = "hsla(0 0% 0% / 75%)";
        this.style.display = "none";
        this.style.height = "100%";
        this.style.left = "0";
        this.style.position = "fixed";
        this.style.top = "0";
        this.style.width = "100%";
        this.style.zIndex = "1";
        // TODO: Difference with ImageViewer
        // this.addEventListener("click", () => {
        //     this.value = null;
        // });

        this._video = this.createChild("video", (e) => {
            // TODO: Difference with ImageViewer
            e.controls = "controls";
            e.style.maxHeight = "100%";
            e.style.objectFit = "contain";
            e.style.width = "100%";
        });

        this._closeButton = this.createChild(Block, (e) => {
            e.innerText = "\u{f00d}";
            e.style.alignContent = "center";
            e.style.color = "white";
            e.style.cursor = "pointer";
            e.style.fontSize = "4rem";
            e.style.fontWeight = "bold";
            e.style.height = "4rem";
            e.style.position = "absolute";
            e.style.right = "0";
            e.style.textAlign = "center";
            e.style.top = "0";
            e.style.width = "4rem";
            // TODO: Difference with ImageViewer
            e.addEventListener("click", () => {
                this.value = null;
            });
        });
    }

    set value(value) {
        if (!value) {
            this.viewedElement = null;
        }

        this.style.display = value ? null : "none";

        if (value) {
            this._video.src = value;
        } else {
            this._video.removeAttribute("src");
        }
    }
}
customElements.define("video-viewer", VideoViewer);

let videoViewer = null;

document.addEventListener("DOMContentLoaded", () => {
    videoViewer = createElement(document.body, VideoViewer);
});

export class VideoBox extends Block {
    constructor() {
        super();

        this.onValueChange = new Signal();

        this.style.alignContent = "center";
        this.style.background = "var(--input-color)";
        this.style.border = "var(--thin-border)";
        this.style.borderRadius = "var(--corners)";
        this.style.height = "100%";
        this.style.overflow = "hidden";
        this.style.position = "relative";
        this.style.textAlign = "center";
        this.style.userSelect = "none";

        this._input = this.createChild("input", (e) => {
            e.type = "file";
            e.accept = "video/*";
            e.style.border = "unset";
            e.style.borderRadius = "unset";
            e.style.height = "100%";
            e.style.minHeight = "10rem";
            e.style.width = "100%";
            e.addEventListener("change", () => {
                let reader = new FileReader();
                reader.addEventListener("load", () => {
                    this.value = reader.result;

                    e.value = null;
                });
                reader.readAsDataURL(e.files[0]);
            });
        });

        this._video = this.createChild("video", (e) => {
            e.style.cursor = "pointer";
            e.style.display = "none";
            e.style.height = "100%";
            e.style.maxWidth = "100%";
            e.style.objectFit = "contain";
            e.style.verticalAlign = "middle";
            e.style.width = "auto";
            e.addEventListener("click", () => {
                videoViewer.value = e.src;
                videoViewer.viewedElement = this;
            });
        });

        this._deleteButton = this.createChild(ToolButton, (e) => {
            e.label = "\u{f00d}";
            e.style.display = "none";
            e.style.position = "absolute";
            e.style.right = "0";
            e.style.top = "0";
            e.onClick.connect(() => {
                this.value = null;
            });
        });
    }

    get height() {
        return this._video.style.minHeight;
    }

    get value() {
        return this._video.src || null;
    }

    set height(value) {
        this._input.style.minHeight = value;
        this._video.style.maxHeight = value;
        this._video.style.minHeight = value;
    }

    set value(value) {
        this._input.style.display = value ? "none" : null;

        if (value) {
            this._video.src = value;
        } else {
            this._video.removeAttribute("src");
        }

        this._video.style.display = value ? null : "none";

        this._deleteButton.style.display = value ? null : "none";

        if (videoViewer.viewedElement == this) {
            videoViewer.value = value;
        }

        this.onValueChange.fire(this.value);
    }
}
customElements.define("video-box", VideoBox);
