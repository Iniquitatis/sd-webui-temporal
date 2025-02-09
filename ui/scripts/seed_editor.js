import {NumberEditor} from "../scripts/base/number_editor.js";
import {ToolButton} from "../scripts/base/tool_button.js";

export class SeedEditor extends NumberEditor {
    constructor() {
        super();

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
