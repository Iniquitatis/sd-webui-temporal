import {NumberBox} from "/scripts/base/number_box.js";
import {Row} from "/scripts/base/row.js";
import {ToolButton} from "/scripts/base/tool_button.js";
import {Signal} from "/scripts/core/signal.js";
import {defineElement} from "/scripts/utils/dom.js";

export class SeedBox extends Row {
    static tag = "ce-seed-box";
    static css = `
        <self> {
            gap: var(--layout-small-gap);
        }
    `;

    constructor() {
        super();

        this.onValueChange = new Signal();

        this._box = this.createChild(NumberBox, (e) => {
            e.minimum = -1;
            e.maximum = 0x7fffffff;
            e.step = 1;
            e.value = -1;
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
defineElement(SeedBox);
