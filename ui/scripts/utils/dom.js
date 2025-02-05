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
