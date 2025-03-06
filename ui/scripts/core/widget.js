import {createElement} from "../../scripts/utils/dom.js";

export class Widget extends HTMLElement {
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
