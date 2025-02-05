import {ToolButton} from "../scripts/core/buttons.js";
import {NumberEditor} from "../scripts/core/value_editors.js";

export class SeedEditor extends NumberEditor {
    constructor() {
        super();

        this.label = "Seed";
        this.variant = "box";
        this.minimum = -1;
        this.maximum = 4294967295;
        this.step = 1;
        this.value = -1;

        this.createChild(ToolButton, (e) => {
            e.label = "\u{1f3b2}";
            e.onClick.connect(() => {
                this.value = Math.floor(Math.random() * (2 ** 32));
            });
        });
    }
}
customElements.define("seed-editor", SeedEditor);
