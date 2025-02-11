export class Signal {
    constructor() {
        this._enabled = true;
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
        if (!this._enabled) return;

        for (let listener of this._listeners) {
            listener(...args);
        }
    }

    withDisabled(callback) {
        this._enabled = false;
        callback();
        this._enabled = true;
    }
}
