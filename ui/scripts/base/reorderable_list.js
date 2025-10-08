import {Accordion} from "/scripts/base/accordion.js";
import {Block} from "/scripts/base/block.js";
import {Column} from "/scripts/base/column.js";
import {Row} from "/scripts/base/row.js";
import {DragController} from "/scripts/core/drag_controller.js";
import {Signal} from "/scripts/core/signal.js";
import {defineElement} from "/scripts/utils/dom.js";

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
    static tag = "ce-reorderable-list";

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
defineElement(ReorderableList);

export class ReorderableAccordion extends Accordion {
    static tag = "ce-reorderable-accordion";

    constructor() {
        super();

        this.createBeforeLabel(Dragger, null, this);
    }
}
defineElement(ReorderableAccordion);

export class ReorderableElement extends Row {
    static tag = "ce-reorderable-element";
    static css = `
        <self> {
            gap: var(--layout-small-gap);
            width: 100%;
        }
    `;

    constructor() {
        super();

        this.createChild(Dragger, null, this);
    }
}
defineElement(ReorderableElement);

class Dragger extends Block {
    static tag = "ce-dragger";
    static css = `
        <self> {
            align-content: center;
            color: var(--hint-color);
            cursor: move;
            height: var(--widget-height);
            max-width: var(--widget-height);
            min-width: var(--widget-height);
            text-align: center;
            user-select: none;
        }
    `;

    constructor(root) {
        super();

        this.dragRoot = root;
        this.innerText = "\u{e410}";
        drag.register(this);
    }
}
defineElement(Dragger);
