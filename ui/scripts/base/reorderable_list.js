import {Accordion} from "../../scripts/base/accordion.js";
import {Block} from "../../scripts/base/block.js";
import {Column} from "../../scripts/base/column.js";
import {Row} from "../../scripts/base/row.js";
import {DragController} from "../../scripts/core/drag_controller.js";
import {Signal} from "../../scripts/core/signal.js";

let drag = new DragController();
drag.onStart.connect((element) => {
    let item = element.dragRoot;
    item.classList.add("dragged");
});
drag.onMove.connect((element, event) => {
    let item = element.dragRoot;
    let list = item.parentElement;

    for (let other of list.childNodes) {
        if (item == other) continue;

        let selfRect = item.getBoundingClientRect();
        let otherRect = other.getBoundingClientRect();

        if (selfRect.top < otherRect.top && event.clientY > otherRect.top) {
            list.insertBefore(other, item);
        }

        if (selfRect.top > otherRect.top && event.clientY < otherRect.bottom) {
            list.insertBefore(item, other);
        }
    }
});
drag.onEnd.connect((element) => {
    let item = element.dragRoot;
    let list = item.parentElement;
    list.onOrderChange.fire();

    item.classList.remove("dragged");
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
            e.dragRoot = this;
            e.innerText = "\u{e410}";
            e.style.alignContent = "center";
            e.style.color = "var(--hint-color)";
            e.style.cursor = "move";
            e.style.height = "var(--widget-height)";
            e.style.maxWidth = "var(--widget-height)";
            e.style.minWidth = "var(--widget-height)";
            e.style.textAlign = "center";
            e.style.userSelect = "none";
            drag.register(e);
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
            e.dragRoot = this;
            e.innerText = "\u{e410}";
            e.style.alignContent = "center";
            e.style.color = "var(--hint-color)";
            e.style.cursor = "move";
            e.style.height = "var(--widget-height)";
            e.style.maxWidth = "var(--widget-height)";
            e.style.minWidth = "var(--widget-height)";
            e.style.textAlign = "center";
            e.style.userSelect = "none";
            drag.register(e);
        });
    }
}
customElements.define("reorderable-element", ReorderableElement);
