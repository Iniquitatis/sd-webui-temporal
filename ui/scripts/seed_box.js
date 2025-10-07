import {NumberBox} from "/scripts/base/number_box.js";
import {Row} from "/scripts/base/row.js";
import {ToolButton} from "/scripts/base/tool_button.js";
import {Signal} from "/scripts/core/signal.js";

export class SeedBox extends Row {
    constructor() {
        super();

        this.onValueChange = new Signal();

        this.style.gap = "var(--layout-small-gap)";

        this._box = this.createChild(NumberBox, (e) => {
            e.minimum = -1;
            e.maximum = 0x7fffffff;
            e.step = 1;
            e.value = -1;
            e.style.width = "100%";
            e.onValueChange.connect((value) => {
                this.onValueChange.fire(value);
            });
        });

        this.createChild(ToolButton, (e) => {
            e.label = "\u{f523}";
            e.onClick.connect(() => {
                this.value = Math.floor(Math.random() * 0x80000000);
            });
        });
    }

    get value() {
        return this._box.value;
    }

    set value(value) {
        this._box.value = value;
    }
}
customElements.define("ce-seed-box", SeedBox);
