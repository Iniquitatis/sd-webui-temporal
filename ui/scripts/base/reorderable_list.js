import {Accordion} from "../../scripts/base/accordion.js";
import {Block} from "../../scripts/base/block.js";
import {Column} from "../../scripts/base/column.js";
import {Row} from "../../scripts/base/row.js";
import {Signal} from "../../scripts/core/signal.js";

let draggedElement = null;

window.addEventListener("touchmove", (event) => {
    if (!draggedElement) return;

    event.stopPropagation();
    event.preventDefault();
}, {passive: false});

window.addEventListener("pointermove", (event) => {
    if (!draggedElement) return;

    event.stopPropagation();

    let list = draggedElement.parentElement;

    for (let other of list.childNodes) {
        if (draggedElement == other) continue;

        let selfRect = draggedElement.getBoundingClientRect();
        let otherRect = other.getBoundingClientRect();

        if (selfRect.top < otherRect.top && event.clientY > otherRect.top) {
            list.insertBefore(other, draggedElement);
        }

        if (selfRect.top > otherRect.top && event.clientY < otherRect.bottom) {
            list.insertBefore(draggedElement, other);
        }
    }
});

window.addEventListener("pointerup", (event) => {
    if (!draggedElement) return;

    event.stopPropagation();

    let list = draggedElement.parentElement;
    list.onOrderChange.fire();

    draggedElement.classList.remove("dragged");

    draggedElement = null;
});

export class ReorderableList extends Column {
    constructor() {
        super();

        this.onOrderChange = new Signal();

        this._mo = new MutationObserver(mutations => {
            mutations.forEach(mutation => {
                if (mutation.type == "childList") {
                    this.visible = this.childElementCount > 0;
                }
            });
        });
        this._mo.observe(this, {childList: true, subtree: false});
    }
}
customElements.define("reorderable-list", ReorderableList);

export class ReorderableAccordion extends Accordion {
    constructor() {
        super();

        this.createBeforeLabel(Block, (e) => {
            e.innerText = "\u{e410}";
            e.style.alignContent = "center";
            e.style.color = "var(--hint-color)";
            e.style.cursor = "move";
            e.style.height = "var(--widget-height)";
            e.style.maxWidth = "var(--widget-height)";
            e.style.minWidth = "var(--widget-height)";
            e.style.textAlign = "center";
            e.style.userSelect = "none";
            e.addEventListener("pointerdown", (event) => {
                event.stopPropagation();

                this.classList.add("dragged");

                draggedElement = this;
            });
        });
    }
}
customElements.define("reorderable-accordion", ReorderableAccordion);

export class ReorderableElement extends Row {
    constructor() {
        super();

        this.style.gap = "var(--layout-small-gap)";
        this.style.width = "100%";

        this.createChild(Block, (e) => {
            e.innerText = "\u{e410}";
            e.style.alignContent = "center";
            e.style.color = "var(--hint-color)";
            e.style.cursor = "move";
            e.style.height = "var(--widget-height)";
            e.style.maxWidth = "var(--widget-height)";
            e.style.minWidth = "var(--widget-height)";
            e.style.textAlign = "center";
            e.style.userSelect = "none";
            e.addEventListener("pointerdown", (event) => {
                event.stopPropagation();

                this.classList.add("dragged");

                draggedElement = this;
            });
        });
    }
}
customElements.define("reorderable-element", ReorderableElement);
