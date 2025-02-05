import {Signal} from "../../scripts/core/signal.js";
import {Widget} from "../../scripts/core/widget.js";
import {createElement} from "../../scripts/utils/dom.js";

export class Row extends Widget {
    constructor() {
        super();

        this.style.display = "flex";
        this.style.flexDirection = "row";
        this.style.gap = "var(--layout-gap)";
    }
}
customElements.define("layout-row", Row);

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

export class Tab extends Widget {
    constructor() {
        super();

        this.name = "Untitled";
    }
}
customElements.define("layout-tab", Tab);

export class Tabs extends Widget {
    constructor() {
        super();

        this._bar = createElement(this, "div", (e) => {
            e.style.display = "flex";
            e.style.flexDirection = "row";
            e.style.flexWrap = "wrap";
            e.style.gap = "calc(var(--layout-gap) / 2)";
            e.style.margin = "0 var(--corners)";
        });

        this._content = createElement(this, "div", (e) => {
            e.style.border = "var(--thin-border)";
            e.style.borderRadius = "var(--corners)";
            e.style.padding = "var(--layout-padding)";
        });
    }

    createTab(name, cls, initializer, ...args) {
        createElement(this._bar, "button", (e) => {
            e.innerText = name;
            e.style.borderBottomLeftRadius = "0";
            e.style.borderBottomRightRadius = "0";
            e.style.fontWeight = "unset";
            e.addEventListener("click", () => {
                this.setActiveTab(name);
            });
        });

        let tab = createElement(this._content, Tab, (e) => {
            e.name = name;
            e.style.display = "none";
        });

        if (this._bar.childElementCount == 1) {
            this.setActiveTab(name);
        }

        return tab.createChild(cls, initializer, ...args);
    }

    setActiveTab(name) {
        for (let child of this._bar.childNodes) {
            if (child.innerText == name) {
                child.classList.add("active");
            }
            else {
                child.classList.remove("active");
            }
        }

        for (let child of this._content.childNodes) {
            if (!(child instanceof Tab)) continue;

            child.style.display = child.name == name ? null : "none";
        }
    }
}
customElements.define("layout-tabs", Tabs);

export class Accordion extends Widget {
    constructor() {
        super();

        this.style.border = "var(--thin-border)";
        this.style.borderRadius = "var(--corners)";

        let callback = () => {
            let visible = this._openToggle.innerText == "\u{25bc}";
            this._content.style.display = visible ? "none" : null;
            this._openToggle.innerText = visible ? "\u{25c0}" : "\u{25bc}";
        };

        this._header = createElement(this, "div", (e) => {
            e.style.alignItems = "center";
            e.style.display = "flex"
            e.style.flexDirection = "row";
            e.style.justifyContent = "space-between";

            this._label = createElement(e, "span", (e) => {
                e.innerText = "Untitled";
                e.style.padding = "var(--padding)";
                e.style.width = "100%";
                e.style.userSelect = "none";
                e.addEventListener("click", callback);
            });

            this._openToggle = createElement(e, "div", (e) => {
                e.innerText = "\u{25c0}";
                e.style.alignContent = "center";
                e.style.cursor = "pointer";
                e.style.fontWeight = "bold";
                e.style.height = "var(--widget-height)";
                e.style.textAlign = "center";
                e.style.userSelect = "none";
                e.style.width = "var(--widget-height)";
                e.addEventListener("click", callback);
            });
        });

        this._content = createElement(this, "div", (e) => {
            e.style.display = "none";
            e.style.padding = "var(--layout-padding)";
        });
    }

    get label() {
        return this._label.innerText;
    }

    set label(value) {
        this._label.innerText = value;
    }

    createChild(cls, initializer, ...args) {
        return createElement(this._content, cls, initializer, ...args);
    }
}
customElements.define("layout-accordion", Accordion);

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
