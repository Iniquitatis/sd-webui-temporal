import {Block} from "/scripts/base/block.js";
import {Widget} from "/scripts/core/widget.js";
import {toggleClass} from "/scripts/utils/dom.js";

class Tab extends Widget {
    constructor() {
        super();

        this.name = "Untitled";
    }
}
customElements.define("ce-tab", Tab);

export class Tabs extends Widget {
    constructor() {
        super();

        this._bar = this.createChild(Block, (e) => {
            e.style.display = "flex";
            e.style.flexDirection = "row";
            e.style.flexWrap = "wrap";
            e.style.gap = "var(--layout-small-gap)";
            e.style.margin = "0 var(--corners)";
        });

        this._content = this.createChild(Block, (e) => {
            e.style.border = "var(--thin-border)";
            e.style.borderRadius = "var(--corners)";
            e.style.padding = "var(--layout-padding)";
        });
    }

    createTab(name, tagOrClass, initializer, ...args) {
        this._bar.createChild("button", (e) => {
            e.innerText = name;
            e.style.borderBottom = "unset";
            e.style.borderBottomLeftRadius = "unset";
            e.style.borderBottomRightRadius = "unset";
            e.style.fontWeight = "unset";
            e.addEventListener("click", () => {
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
customElements.define("ce-tabs", Tabs);
