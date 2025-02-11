import {Block} from "../../scripts/base/block.js";
import {ValueEditor} from "../../scripts/base/value_editor.js";
import {Signal} from "../../scripts/core/signal.js";
import {clearElement} from "../../scripts/utils/dom.js";

export class Radio extends ValueEditor {
    constructor() {
        super();

        this.onValueChange = new Signal();

        this._choices = {};
        this._value = null;

        this._buttons = this._content.createChild(Block, (e) => {
            e.style.display = "flex";
            e.style.flexDirection = "column";
            e.style.gap = "var(--layout-gap)";
            e.style.width = "100%";
        });
    }

    get choices() {
        return this._choices;
    }

    get value() {
        return this._value;
    }

    set choices(value) {
        this._choices = value;

        clearElement(this._buttons);

        for (let [key, name] of Object.entries(this._choices)) {
            this._buttons.createChild(Block, (e) => {
                e.style.alignItems = "center";
                e.style.display = "flex";
                e.style.flexDirection = "row";
                e.style.gap = "calc(var(--horizontal-padding) / 2)";

                let callback = (event) => {
                    this._value = event.target.key;
                    this._updateButtons();

                    this.onValueChange.fire(this._value);
                };

                e.createChild("input", (e) => {
                    e.key = key;
                    e.type = "radio";
                    e.addEventListener("click", callback);
                });

                e.createChild("label", (e) => {
                    e.key = key;
                    e.innerText = name;
                    e.addEventListener("click", callback);
                });
            });
        }

        if (!this._value) {
            this._value = Object.keys(this._choices)[0];
        }
    }

    set value(value) {
        this._value = value;
        this._updateButtons();

        this.onValueChange.fire(value);
    }

    _updateButtons() {
        for (let button of this._buttons.querySelectorAll("input")) {
            button.checked = button.key == this._value;
        }
    }
}
customElements.define("custom-radio", Radio);
