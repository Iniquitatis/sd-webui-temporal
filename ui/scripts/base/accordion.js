import {Widget} from "../../scripts/core/widget.js";
import {createElement} from "../../scripts/utils/dom.js";

export class Accordion extends Widget {
    constructor() {
        super();

        this.style.border = "var(--thin-border)";
        this.style.borderRadius = "var(--corners)";

        let callback = () => {
            let visible = this._openToggle.innerText == "\u{25bc}";
            this._content.style.display = visible ? "none" : null;
            this._openToggle.innerText = visible ? "\u{25c0}" : "\u{25bc}";
        };

        this._header = createElement(this, "div", (e) => {
            e.style.alignItems = "center";
            e.style.display = "flex"
            e.style.flexDirection = "row";
            e.style.justifyContent = "space-between";

            this._label = createElement(e, "span", (e) => {
                e.innerText = "Untitled";
                e.style.padding = "var(--padding)";
                e.style.width = "100%";
                e.style.userSelect = "none";
                e.addEventListener("click", callback);
            });

            this._openToggle = createElement(e, "div", (e) => {
                e.innerText = "\u{25c0}";
                e.style.alignContent = "center";
                e.style.cursor = "pointer";
                e.style.fontWeight = "bold";
                e.style.height = "var(--widget-height)";
                e.style.textAlign = "center";
                e.style.userSelect = "none";
                e.style.width = "var(--widget-height)";
                e.addEventListener("click", callback);
            });
        });

        this._content = createElement(this, "div", (e) => {
            e.style.display = "none";
            e.style.padding = "var(--layout-padding)";
        });
    }

    get label() {
        return this._label.innerText;
    }

    set label(value) {
        this._label.innerText = value;
    }

    createChild(cls, initializer, ...args) {
        return createElement(this._content, cls, initializer, ...args);
    }
}
customElements.define("layout-accordion", Accordion);
