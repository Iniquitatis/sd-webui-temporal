import {Accordion} from "../../scripts/base/accordion.js";
import {Block} from "../../scripts/base/block.js";
import {Column} from "../../scripts/base/column.js";
import {Signal} from "../../scripts/core/signal.js";
import {createElement} from "../../scripts/utils/dom.js";

let draggedAccordion = null;

window.addEventListener("touchmove", (event) => {
    if (!draggedAccordion) return;

    event.stopPropagation();
    event.preventDefault();
}, {passive: false});

window.addEventListener("pointermove", (event) => {
    if (!draggedAccordion) return;

    event.stopPropagation();

    let list = draggedAccordion.parentElement;

    for (let other of list.childNodes) {
        if (draggedAccordion == other) continue;

        let selfRect = draggedAccordion.getBoundingClientRect();
        let otherRect = other.getBoundingClientRect();

        if (selfRect.top < otherRect.top && event.clientY > otherRect.top) {
            list.insertBefore(other, draggedAccordion);
        }

        if (selfRect.top > otherRect.top && event.clientY < otherRect.bottom) {
            list.insertBefore(draggedAccordion, other);
        }
    }
});

window.addEventListener("pointerup", (event) => {
    if (!draggedAccordion) return;

    event.stopPropagation();

    let list = draggedAccordion.parentElement;
    list.onOrderChange.fire();

    draggedAccordion.classList.remove("dragged");

    draggedAccordion = null;
});

export class ReorderableList extends Column {
    constructor() {
        super();

        this.onOrderChange = new Signal();
    }
}
customElements.define("reorderable-list", ReorderableList);

export class ReorderableAccordion extends Accordion {
    constructor() {
        super();

        this._header.insertBefore(createElement(null, Block, (e) => {
            e.innerText = ":::";
            e.style.alignContent = "center";
            e.style.color = "var(--hint-color)";
            e.style.cursor = "move";
            e.style.height = "var(--widget-height)";
            e.style.textAlign = "center";
            e.style.userSelect = "none";
            e.style.width = "calc(var(--widget-height) * 1.2)";
            e.addEventListener("pointerdown", (event) => {
                event.stopPropagation();

                this.classList.add("dragged");

                draggedAccordion = this;
            });
        }), this._header.firstChild);
    }
}
customElements.define("reorderable-accordion", ReorderableAccordion);
