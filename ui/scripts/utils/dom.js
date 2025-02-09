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
