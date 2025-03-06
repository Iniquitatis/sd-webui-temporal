import {createElement} from "../../scripts/utils/dom.js";

export class Widget extends HTMLElement {
    get visible() {
        return this.classList.contains("hidden");
    }

    set visible(value) {
        if (value) {
            this.classList.add("hidden");
        } else {
            this.classList.remove("hidden");
        }
    }

    attachTitle(title) {}

    canAttachTitle() {
        return false;
    }

    createChild(tagOrClass, initializer, ...args) {
        return createElement(this, tagOrClass, initializer, ...args);
    }

    isComplexWidget() {
        return false;
    }
}
