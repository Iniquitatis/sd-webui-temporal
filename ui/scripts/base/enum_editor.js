import {ValueEditor} from "../../scripts/base/value_editor.js";
import {Signal} from "../../scripts/core/signal.js";
import {createElement} from "../../scripts/utils/dom.js";

export class EnumEditor extends ValueEditor {
    constructor() {
        super();

        this._choices = {};

        this._select = createElement(this._content, "select", (e) => {
            e.style.width = "100%";
            e.addEventListener("change", () => {
                this.onValueChange.fire(this.value);
            });
        });

        // NOTE: Intentionally disconnected
        this._radio = createElement(null, "fieldset", (e) => {
            e.style.display = "flex";
            e.style.flexDirection = "column";
            e.style.gap = "calc(var(--layout-gap) / 2)";
            e.style.width = "100%";

            createElement(e, "legend", (e) => {
                e.innerText = "FIXME";
            });
        });

        this.onValueChange = new Signal();
    }

    get choices() {
        return this._choices;
    }

    get value() {
        return [...Object.keys(this._choices)][this._select.selectedIndex];
    }

    get variant() {
        return this._content.contains(this._select) ? "menu" : "radio";
    }

    set choices(value) {
        this._choices = value;

        while (this._select.contains(this._select.firstChild)) {
            this._select.removeChild(this._select.firstChild);
        }

        for (let name of Object.values(this._choices)) {
            createElement(this._select, "option", (e) => {
                e.label = name;
            });

            createElement(this._radio, "div", (e) => {
                createElement(e, "input", (e) => {
                    e.type = "radio";
                });

                createElement(e, "label", (e) => {
                    e.innerText = name;
                });
            });
        }

        if (this._select.selectedIndex == -1 && Object.keys(value).length > 0) {
            this._select.selectedIndex = 0;
        }
    }

    set value(value) {
        this._select.selectedIndex = value ? [...Object.keys(this._choices)].indexOf(value) : 0;

        this.onValueChange.fire(this.value);
    }

    set variant(value) {
        if (value == "radio" && this._content.contains(this._select) && !this._content.contains(this._radio)) {
            this._content.removeChild(this._select);
            this._content.appendChild(this._radio);
        } else if (this._content.contains(this._radio) && !this._content.contains(this._select)) {
            this._content.removeChild(this._radio);
            this._content.appendChild(this._select);
        }
    }
}
customElements.define("enum-editor", EnumEditor);
