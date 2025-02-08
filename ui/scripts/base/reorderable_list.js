import {Accordion} from "../../scripts/base/accordion.js";
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

    let parent = draggedAccordion.parentElement;

    parent.accordions.forEach((other) => {
        if (draggedAccordion == other) return;

        let selfRect = draggedAccordion.getBoundingClientRect();
        let otherRect = other.getBoundingClientRect();

        if (selfRect.top < otherRect.top && event.clientY > otherRect.top) {
            parent.insertBefore(other, draggedAccordion);
        }

        if (selfRect.top > otherRect.top && event.clientY < otherRect.bottom) {
            parent.insertBefore(draggedAccordion, other);
        }
    })
});

window.addEventListener("pointerup", (event) => {
    if (!draggedAccordion) return;

    event.stopPropagation();

    let list = draggedAccordion.parentElement;
    list.onOrderChange.fire(list.order);

    draggedAccordion.classList.remove("dragged");

    draggedAccordion = null;
});

export class ReorderableList extends Column {
    constructor() {
        super();

        this.onOrderChange = new Signal();
    }

    get accordions() {
        return [...this.childNodes].filter((node) => node instanceof ReorderableAccordion);
    }

    get order() {
        return [...this.childNodes].map((node) => node.key);
    }
}
customElements.define("layout-reorderable-list", ReorderableList);

export class ReorderableAccordion extends Accordion {
    constructor(key) {
        super();

        this.key = key;

        this._header.insertBefore(createElement(null, "span", (e) => {
            e.innerText = ":::";
            e.style.alignContent = "center";
            e.style.color = "var(--hint-color)";
            e.style.cursor = "move";
            e.style.height = "var(--widget-height)";
            e.style.textAlign = "center";
            e.style.userSelect = "none";
            e.style.width = "var(--widget-height)";
            e.addEventListener("pointerdown", (event) => {
                event.stopPropagation();

                this.classList.add("dragged");

                draggedAccordion = this;
            });
        }), this._header.firstChild);
    }
}
customElements.define("layout-reorderable-accordion", ReorderableAccordion);
