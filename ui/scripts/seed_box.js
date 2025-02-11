import {NumberBox} from "../scripts/base/number_box.js";
import {ToolButton} from "../scripts/base/tool_button.js";

export class SeedBox extends NumberBox {
    constructor() {
        super();

        this.minimum = -1;
        this.maximum = 4294967295;
        this.step = 1;
        this.value = -1;

        this.createChild(ToolButton, (e) => {
            e.label = "\u{1f3b2}\u{fe0e}";
            e.onClick.connect(() => {
                this.value = Math.floor(Math.random() * (2 ** 32));
            });
        });
    }
}
customElements.define("seed-box", SeedBox);
