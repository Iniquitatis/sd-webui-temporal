import {dedent} from "/scripts/utils/string.js";
import {iterateAncestry} from "/scripts/utils/types.js";

export function clearElement(element) {
    while (element.contains(element.firstChild)) {
        element.removeChild(element.firstChild);
    }
}

export function createElement(parent, tagOrClass, initializer, ...args) {
    let result = typeof tagOrClass === "string" ?
        document.createElement(tagOrClass, ...args) :
        new tagOrClass(...args);

    if (initializer) {
        initializer(result);
    }

    if (parent) {
        parent.appendChild(result);
    }

    return result;
}

export function defineElement(cls) {
    if (!cls.tag) return;

    let parts = Array.from(iterateAncestry(cls, HTMLElement))
        .filter((x) => Object.hasOwn(x, "css") && x.css)
        .map((x) => dedent(x.css).replaceAll("<self>", cls.tag));

    if (parts.length > 0) {
        let style = document.createElement("style");
        style.textContent = parts.join("\n\n");
        document.head.appendChild(style);
    }

    customElements.define(cls.tag, cls);
}

export function toggleClass(element, className, enabled) {
    if (enabled) {
        element.classList.add(className);
    } else {
        element.classList.remove(className);
    }
}
