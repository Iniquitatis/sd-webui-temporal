export class Signal {
    constructor() {
        this._listeners = [];
    }

    connect(func) {
        this._listeners.push(func);
        return func;
    }

    disconnect(func) {
        this._listeners.splice(this._listeners.indexOf(func), 1);
    }

    fire(...args) {
        for (let listener of this._listeners) {
            listener(...args);
        }
    }
}
