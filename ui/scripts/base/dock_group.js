import {Block} from "/scripts/base/block.js";
import {Column} from "/scripts/base/column.js";
import {Row} from "/scripts/base/row.js";
import {ToolButton} from "/scripts/base/tool_button.js";
import {Widget} from "/scripts/core/widget.js";
import {defineElement} from "/scripts/utils/dom.js";

class Dock extends Column {
    static tag = "ce-dock";
    static css = `
        <self> {
            background: var(--background-color);
            border-right: var(--thin-border);
            height: 100%;
            max-width: 100%;
            padding: var(--layout-padding);
            width: 40rem;
        }

        <self> > ce-row > ce-block {
            align-content: center;
            color: var(--hint-color);
            font-size: var(--title-size);
        }

        <self> > ce-block {
            overflow-y: auto;
        }
    `;

    constructor() {
        super();

        super.createChild(Row, (e) => {
            this._label = e.createChild(Block);

            e.createChild(ToolButton, (e) => {
                e.label = "\u{f323}";
                e.onClick.connect(() => {
                    this.visible = false;
                });
            });
        });

        this._content = super.createChild(Block);
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
defineElement(Dock);

export class DockGroup extends Widget {
    static tag = "ce-dock-group";
    static css = `
        <self> {
            display: flex;
            flex-direction: row;
            inset: 0;
            max-width: 100%;
            pointer-events: none;
            position: absolute;
            z-index: 1;
        }

        <self> > ce-block {
            max-width: 100%;
        }

        <self> > ce-block > ce-dock {
            pointer-events: auto;
        }

        <self> > ce-column {
            gap: var(--layout-small-gap);
            margin: var(--layout-small-gap) 0;
        }

        <self> > ce-column > ce-tool-button {
            border-bottom-left-radius: unset;
            border-left: unset;
            border-top-left-radius: unset;
            height: calc(var(--widget-height) * 1.5);
            pointer-events: auto;
        }
    `;

    constructor() {
        super();

        this._content = this.createChild(Block);

        this._buttons = this.createChild(Column);
    }

    createDock(icon, label, tagOrClass, initializer, ...args) {
        let dock = this._content.createChild(Dock, (e) => {
            e.label = label;
            e.visible = false;
        });

        this._buttons.createChild(ToolButton, (e) => {
            e.label = icon;
            e.onClick.connect(() => {
                this.setActiveDock(label);
            });
        });

        return dock.createChild(tagOrClass, initializer, ...args);
    }

    setActiveDock(label) {
        for (let child of this._content.childNodes) {
            child.visible = child.label == label;
        }
    }
}
defineElement(DockGroup);
