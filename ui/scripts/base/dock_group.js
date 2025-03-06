import {Block} from "../../scripts/base/block.js";
import {Column} from "../../scripts/base/column.js";
import {Row} from "../../scripts/base/row.js";
import {ToolButton} from "../../scripts/base/tool_button.js";
import {Widget} from "../../scripts/core/widget.js";

class Dock extends Column {
    constructor() {
        super();

        this.style.background = "var(--background-color)";
        this.style.borderRight = "var(--thin-border)";
        this.style.height = "100%";
        this.style.maxWidth = "100%";
        this.style.padding = "var(--layout-padding)";
        this.style.width = "40rem";

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

export class DockGroup extends Widget {
    constructor() {
        super();

        this.style.display = "flex";
        this.style.flexDirection = "row";
        this.style.inset = "0";
        this.style.maxWidth = "100%";
        this.style.pointerEvents = "none";
        this.style.position = "absolute";
        this.style.zIndex = "1";

        this._content = this.createChild(Block, (e) => {
            e.style.maxWidth = "100%";
        });

        this._buttons = this.createChild(Column, (e) => {
            e.style.gap = "calc(var(--layout-gap) / 2)";
            e.style.margin = "calc(var(--layout-gap) / 2) 0";
        });
    }

    createDock(icon, label, tagOrClass, initializer, ...args) {
        let dock = this._content.createChild(Dock, (e) => {
            e.label = label;
            e.style.display = "none";
            e.style.pointerEvents = "auto";
        });

        this._buttons.createChild(ToolButton, (e) => {
            e.label = icon;
            e.style.height = "calc(var(--widget-height) * 1.5)";
            e.style.pointerEvents = "auto";
            e._button.style.borderBottomLeftRadius = "unset";
            e._button.style.borderLeft = "unset";
            e._button.style.borderTopLeftRadius = "unset";
            e.onClick.connect(() => {
                this.setActiveDock(label);
            });
        });

        return dock.createChild(tagOrClass, initializer, ...args);
    }

    setActiveDock(label) {
        for (let child of this._content.childNodes) {
            child.style.display = child.label == label ? "flex" : "none";
        }
    }
}
customElements.define("dock-group", DockGroup);
