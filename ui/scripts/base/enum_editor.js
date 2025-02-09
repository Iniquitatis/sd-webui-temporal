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

        this.onValueChange = new Signal();
    }

    get choices() {
        return this._choices;
    }

    get value() {
        return [...Object.keys(this._choices)][this._select.selectedIndex];
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
        }

        if (this._select.selectedIndex == -1 && Object.keys(value).length > 0) {
            this._select.selectedIndex = 0;
        }
    }

    set value(value) {
        this._select.selectedIndex = value ? [...Object.keys(this._choices)].indexOf(value) : 0;

        this.onValueChange.fire(this.value);
    }
}
customElements.define("enum-editor", EnumEditor);
