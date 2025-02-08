import {Widget} from "../../scripts/core/widget.js";
import {createElement} from "../../scripts/utils/dom.js";

class Tab extends Widget {
    constructor() {
        super();

        this.name = "Untitled";
    }
}
customElements.define("layout-tab", Tab);

export class Tabs extends Widget {
    constructor() {
        super();

        this._bar = createElement(this, "div", (e) => {
            e.style.display = "flex";
            e.style.flexDirection = "row";
            e.style.flexWrap = "wrap";
            e.style.gap = "calc(var(--layout-gap) / 2)";
            e.style.margin = "0 var(--corners)";
        });

        this._content = createElement(this, "div", (e) => {
            e.style.border = "var(--thin-border)";
            e.style.borderRadius = "var(--corners)";
            e.style.padding = "var(--layout-padding)";
        });
    }

    createTab(name, cls, initializer, ...args) {
        createElement(this._bar, "button", (e) => {
            e.innerText = name;
            e.style.borderBottomLeftRadius = "0";
            e.style.borderBottomRightRadius = "0";
            e.style.fontWeight = "unset";
            e.addEventListener("click", () => {
                this.setActiveTab(name);
            });
        });

        let tab = createElement(this._content, Tab, (e) => {
            e.name = name;
            e.style.display = "none";
        });

        if (this._bar.childElementCount == 1) {
            this.setActiveTab(name);
        }

        return tab.createChild(cls, initializer, ...args);
    }

    setActiveTab(name) {
        for (let child of this._bar.childNodes) {
            if (child.innerText == name) {
                child.classList.add("active");
            }
            else {
                child.classList.remove("active");
            }
        }

        for (let child of this._content.childNodes) {
            if (!(child instanceof Tab)) continue;

            child.style.display = child.name == name ? null : "none";
        }
    }
}
customElements.define("layout-tabs", Tabs);
