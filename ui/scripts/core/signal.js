export class Signal {
    constructor() {
        this._locks = 0;
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
        if (this._locks > 0) return;

        for (let listener of this._listeners) {
            listener(...args);
        }
    }

    withDisabled(callback) {
        this._locks += 1;
        callback();
        this._locks -= 1;
    }
}
