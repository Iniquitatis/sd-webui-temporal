import {ValueEditor} from "../../scripts/base/value_editor.js";
import {Signal} from "../../scripts/core/signal.js";
import {clearElement, createElement} from "../../scripts/utils/dom.js";

export class Radio extends ValueEditor {
    constructor() {
        super();

        this._choices = {};

        this._radio = createElement(this._content, "fieldset", (e) => {
            e.style.width = "100%";

            createElement(e, "legend", (e) => {
                e.innerText = "FIXME";
            });

            this._buttons = createElement(e, "div", (e) => {
                e.style.display = "flex";
                e.style.flexDirection = "column";
                e.style.gap = "calc(var(--layout-gap) / 2)";
            });
        });

        this.onValueChange = new Signal();
    }

    // TODO: From here and onwards
    get choices() {
        return this._choices;
    }

    get value() {
        return [...Object.keys(this._choices)][this._select.selectedIndex];
    }

    set choices(value) {
        this._choices = value;

        clearElement(this._buttons);

        for (let name of Object.values(this._choices)) {
            createElement(this._buttons, "div", (e) => {
                createElement(e, "input", (e) => {
                    e.type = "radio";
                });

                createElement(e, "label", (e) => {
                    e.innerText = name;
                });
            });
        }

        // if (this._select.selectedIndex == -1 && Object.keys(value).length > 0) {
        //     this._select.selectedIndex = 0;
        // }
    }

    set value(value) {
        // this._select.selectedIndex = value ? [...Object.keys(this._choices)].indexOf(value) : 0;

        this.onValueChange.fire(this.value);
    }
}
customElements.define("custom-radio", Radio);
