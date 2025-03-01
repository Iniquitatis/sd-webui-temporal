import {Block} from "../../scripts/base/block.js";
import {Column} from "../../scripts/base/column.js";
import {Row} from "../../scripts/base/row.js";
import {ToolButton} from "../../scripts/base/tool_button.js";

export class Dock extends Column {
    constructor() {
        super();

        this.style.background = "var(--background-color)";
        this.style.borderRight = "var(--thin-border)";
        this.style.height = "100%";
        this.style.maxWidth = "100%";
        this.style.padding = "var(--layout-padding)";
        this.style.position = "absolute";
        this.style.width = "40rem";
        this.style.zIndex = "1";

        super.createChild(Row, (e) => {
            this._label = e.createChild(Block, (e) => {
                e.style.alignContent = "center";
                e.style.color = "var(--hint-color)";
                e.style.fontSize = "1.2rem";
                e.style.width = "100%";
            });

            e.createChild(ToolButton, (e) => {
                e.label = "\u{f323}";
                e.onClick.connect(() => {
                    this.style.display = "none";
                });
            });
        });

        this._content = super.createChild(Block, (e) => {
            e.style.overflowY = "auto";
        });
    }

    get label() {
        return this._label.innerText;
    }

    set label(value) {
        this._label.innerText = value;
    }

    createChild(tagOrClass, initializer, ...args) {
        return this._content.createChild(tagOrClass, initializer, ...args);
    }
}
customElements.define("dock-widget", Dock);
