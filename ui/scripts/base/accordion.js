import {Block} from "../../scripts/base/block.js";
import {MultiStateToggle} from "../../scripts/base/multi_state_toggle.js";
import {Widget} from "../../scripts/core/widget.js";

export class Accordion extends Widget {
    constructor() {
        super();

        this.style.border = "var(--thin-border)";
        this.style.borderRadius = "var(--corners)";

        this._header = super.createChild(Block, (e) => {
            e.style.alignItems = "center";
            e.style.display = "flex"
            e.style.flexDirection = "row";
            e.style.justifyContent = "space-between";

            this._label = e.createChild(Block, (e) => {
                e.style.alignContent = "center";
                e.style.height = "var(--widget-height)";
                e.style.padding = "0 var(--horizontal-padding)";
                e.style.userSelect = "none";
                e.style.width = "100%";
                e.addEventListener("click", () => {
                    this._openToggle.nextState();
                });
            });

            this._openToggle = e.createChild(MultiStateToggle, (e) => {
                e.states = {opened: "\u{f0d7}", closed: "\u{f0d9}"};
                e.value = "closed";
                e.onValueChange.connect((value) => {
                    this._content.style.display = value == "opened" ? "block" : "none";
                });
            });
        });

        this._content = super.createChild(Block, (e) => {
            e.style.display = "none";
            e.style.padding = "var(--layout-padding)";
            e.style.paddingTop = "0";
        });
    }

    get label() {
        return this._label.innerText;
    }

    set label(value) {
        this._label.innerText = value;
    }

    createChild(tagOrClass, initializer, ...args) {
        return this._content.createChild(tagOrClass, initializer, ...args);
    }
}
customElements.define("layout-accordion", Accordion);
