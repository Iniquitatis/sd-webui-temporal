import {Block} from "../../scripts/base/block.js";
import {ToolButton} from "../../scripts/base/tool_button.js";
import {ValueEditor} from "../../scripts/base/value_editor.js";
import {Signal} from "../../scripts/core/signal.js";
import {Widget} from "../../scripts/core/widget.js";
import {createElement} from "../../scripts/utils/dom.js";

// FIXME: Deduplicate with VideoViewer
class ImageViewer extends Widget {
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
        this.addEventListener("click", () => {
            this.value = null;
        });

        this._img = this.createChild("img", (e) => {
            e.style.objectFit = "contain";
            e.style.width = "100%";
        });

        this._closeButton = this.createChild(Block, (e) => {
            e.innerText = "\u{274c}\u{fe0e}";
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
        });
    }

    set value(value) {
        if (!value) {
            this.viewedElement = null;
        }

        this.style.display = value ? null : "none";

        this._img.src = value ?? undefined;
    }
}
customElements.define("image-viewer", ImageViewer);

let imageViewer = null;

document.addEventListener("DOMContentLoaded", () => {
    imageViewer = createElement(document.body, ImageViewer);
});

export class ImageBox extends ValueEditor {
    constructor() {
        super();

        this.onValueChange = new Signal();

        this._content.style.height = "calc(100% - var(--widget-height))";

        this._content.createChild(Block, (e) => {
            e.style.alignContent = "center";
            e.style.border = "var(--thin-border)";
            e.style.borderRadius = "var(--corners)";
            e.style.minHeight = "10rem";
            e.style.position = "relative";
            e.style.textAlign = "center";
            e.style.userSelect = "none";
            e.style.width = "100%";

            this._input = e.createChild("input", (e) => {
                e.type = "file";
                e.accept = "image/*";
                e.style.height = "100%";
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

            this._img = e.createChild("img", (e) => {
                e.style.cursor = "pointer";
                e.style.display = "none";
                e.style.height = "100%";
                e.style.maxWidth = "100%";
                e.style.objectFit = "contain";
                e.addEventListener("click", () => {
                    imageViewer.value = e.src;
                    imageViewer.viewedElement = this;
                });
            });

            this._deleteButton = e.createChild(ToolButton, (e) => {
                e.label = "\u{274c}\u{fe0e}";
                e.style.display = "none";
                e.style.position = "absolute";
                e.style.right = "0";
                e.style.top = "0";
                e.onClick.connect(() => {
                    this.value = null;
                });
            });
        });
    }

    get height() {
        return this.style.height;
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

        this._deleteButton.style.display = value ? null : "none";

        if (imageViewer.viewedElement == this) {
            imageViewer.value = value;
        }

        this.onValueChange.fire(this.value);
    }
}
customElements.define("image-box", ImageBox);
