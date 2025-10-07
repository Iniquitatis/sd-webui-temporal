import {Widget} from "/scripts/core/widget.js";

export class Row extends Widget {
    constructor() {
        super();

        this.style.display = "flex";
        this.style.flexDirection = "row";
        this.style.gap = "var(--layout-gap)";
    }
}
customElements.define("ce-row", Row);
