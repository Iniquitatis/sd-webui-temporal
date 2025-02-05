export class Widget extends HTMLElement {
    createChild(cls, initializer, ...args) {
        let elem = new cls(...args);

        if (initializer) {
            initializer(elem);
        }

        this.appendChild(elem);

        return elem;
    }
}
