import {createElement} from "../../scripts/utils/dom.js";

export class Widget extends HTMLElement {
    createChild(tagOrClass, initializer, ...args) {
        return createElement(this, tagOrClass, initializer, ...args);
    }
}
