import {Column} from "../../scripts/base/column.js";
import {Dock} from "../../scripts/base/dock.js";
import {ToolButton} from "../../scripts/base/tool_button.js";
import {Widget} from "../../scripts/core/widget.js";

export class DockGroup extends Widget {
    constructor() {
        super();

        this._buttons = this.createChild(Column, (e) => {
            e.style.inset = "0";
            e.style.pointerEvents = "none";
            e.style.position = "absolute";
            e.style.zIndex = "1";
        });
    }

    createDock(icon, label, tagOrClass, initializer, ...args) {
        let dock = this.createChild(Dock, (e) => {
            e.label = label;
            e.style.display = "none";
        });

        this._buttons.createChild(ToolButton, (e) => {
            e.label = icon;
            e.style.pointerEvents = "auto";
            e._button.style.borderBottomLeftRadius = "unset";
            e._button.style.borderLeft = "unset";
            e._button.style.borderTopLeftRadius = "unset";
            e.onClick.connect(() => {
                dock.style.display = "flex";
            });
        });

        return dock.createChild(tagOrClass, initializer, ...args);
    }
}
customElements.define("dock-group", DockGroup);
