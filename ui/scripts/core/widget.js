import {createElement, toggleClass} from "/scripts/utils/dom.js";

export class Widget extends HTMLElement {
    get enabled() {
        return !this.classList.contains("disabled");
    }

    get visible() {
        return !this.classList.contains("hidden");
    }

    set enabled(value) {
        this.toggleClass("disabled", !value);
    }

    set visible(value) {
        this.toggleClass("hidden", !value);
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

    toggleClass(className, enabled) {
        toggleClass(this, className, enabled);
    }
}
