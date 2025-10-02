import {Widget} from "/scripts/core/widget.js";

export class Block extends Widget {
    constructor() {
        super();

        this.style.display = "block";
    }
}
customElements.define("custom-block", Block);
