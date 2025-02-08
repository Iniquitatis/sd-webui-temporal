import {Widget} from "../../scripts/core/widget.js";

export class Column extends Widget {
    constructor() {
        super();

        this.style.display = "flex";
        this.style.flexDirection = "column";
        this.style.gap = "var(--layout-gap)";
        this.style.width = "100%";
    }
}
customElements.define("layout-column", Column);
