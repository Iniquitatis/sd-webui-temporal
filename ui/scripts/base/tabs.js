import {Block} from "/scripts/base/block.js";
import {Button} from "/scripts/base/button.js";
import {Widget} from "/scripts/core/widget.js";
import {defineElement, toggleClass} from "/scripts/utils/dom.js";

class Tab extends Widget {
    static tag = "ce-tab";

    constructor() {
        super();

        this.name = "Untitled";
    }
}
defineElement(Tab);

export class Tabs extends Widget {
    static tag = "ce-tabs";
    static css = `
        <self> > ce-block:nth-of-type(1) {
            display: flex;
            flex-direction: row;
            flex-wrap: wrap;
            gap: var(--layout-small-gap);
            margin: 0 var(--corners);
        }

        <self> > ce-block:nth-of-type(1) > ce-button {
            border-bottom: unset;
            border-bottom-left-radius: unset;
            border-bottom-right-radius: unset;
            font-weight: unset;
        }

        <self> > ce-block:nth-of-type(2) {
            border: var(--thin-border);
            border-radius: var(--corners);
            padding: var(--layout-padding);
        }
    `;

    constructor() {
        super();

        this._bar = this.createChild(Block);

        this._content = this.createChild(Block);
    }

    createTab(name, tagOrClass, initializer, ...args) {
        this._bar.createChild(Button, (e) => {
            e.innerText = name;
            e.onClick.connect(() => {
                this.setActiveTab(name);
            });
        });

        let tab = this._content.createChild(Tab, (e) => {
            e.name = name;
            e.visible = false;
        });

        if (this._bar.childElementCount == 1) {
            this.setActiveTab(name);
        }

        return tab.createChild(tagOrClass, initializer, ...args);
    }

    setActiveTab(name) {
        for (let child of this._bar.childNodes) {
            toggleClass(child, "active", child.innerText == name);
        }

        for (let child of this._content.childNodes) {
            child.visible = child.name == name;
        }
    }
}
defineElement(Tabs);
