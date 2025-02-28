import {Block} from "../../scripts/base/block.js";
import {Row} from "../../scripts/base/row.js";
import {ToolButton} from "../../scripts/base/tool_button.js";
import {FilePicker} from "../../scripts/core/file_picker.js";
import {Signal} from "../../scripts/core/signal.js";
import {Widget} from "../../scripts/core/widget.js";
import {createElement} from "../../scripts/utils/dom.js";

class MediaViewer extends Widget {
    constructor(controller, element) {
        super();

        this.style.alignContent = "center";
        this.style.background = "hsla(0 0% 0% / 75%)";
        this.style.height = "100%";
        this.style.left = "0";
        this.style.position = "fixed";
        this.style.top = "0";
        this.style.width = "100%";
        this.style.zIndex = "1";

        this._element = this.appendChild(element.cloneNode());

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
            e.addEventListener("click", () => {
                this.parentElement.removeChild(this);
                controller._viewer = null;
            });
        });
    }

    get value() {
        return this._element.value;
    }

    set value(value) {
        this._element.value = value;
    }
}
customElements.define("media-viewer", MediaViewer);

export class MediaBox extends Block {
    constructor(cls, mimeType = "*/*") {
        super();

        this.onValueChange = new Signal();

        this._filePicker = new FilePicker();
        this._filePicker.mimeType = mimeType;
        this._filePicker.onLoad.connect((data) => {
            this.value = data;
        });

        this.style.alignContent = "center";
        this.style.background = "var(--input-color)";
        this.style.border = "var(--thin-border)";
        this.style.borderRadius = "var(--corners)";
        this.style.height = "100%";
        this.style.minHeight = "150px";
        this.style.overflow = "hidden";
        this.style.position = "relative";
        this.style.textAlign = "center";
        this.style.userSelect = "none";
        this.addEventListener("click", () => {
            if (!this.value) {
                this._filePicker.open();
            }
        });

        this._element = this.createChild(cls, (e) => {
            e.style.height = "100%";
            e.onValueChange.connect((value) => {
                this._buttonRow.style.display = value ? "flex" : "none";

                if (this._viewer) {
                    this._viewer.value = value;
                }

                this.onValueChange.fire(this.value);
            });
        });

        this._buttonRow = this.createChild(Row, (e) => {
            e.style.display = "none";
            e.style.flexDirection = "row-reverse";
            e.style.position = "absolute";
            e.style.right = "var(--layout-padding)";
            e.style.top = "var(--layout-padding)";
            e.style.width = "auto";

            e.createChild(ToolButton, (e) => {
                e.label = "\u{f00d}";
                e.onClick.connect(() => {
                    this.value = null;
                });
            });

            e.createChild(ToolButton, (e) => {
                e.label = "\u{f065}";
                e.onClick.connect(() => {
                    this._viewer = createElement(document.body, MediaViewer, (e) => {
                        e.value = this.value;
                    }, this, this._element);
                });
            });

            e.createChild(ToolButton, (e) => {
                e.label = "\u{f093}";
                e.onClick.connect(() => {
                    this._filePicker.open();
                });
            });
        });

        this._viewer = null;
    }

    get height() {
        return this.style.minHeight;
    }

    get value() {
        return this._element.value;
    }

    set height(value) {
        this.style.minHeight = value;
    }

    set value(value) {
        this._element.value = value;
    }
}
customElements.define("media-box", MediaBox);
