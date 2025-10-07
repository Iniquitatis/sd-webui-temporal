export class Timer {
    constructor(callback, interval) {
        this._callback = callback;
        this._interval = interval;
        this._instance = null;
    }

    start() {
        if (this._instance) return;

        this._instance = window.setInterval(this._callback, this._interval * 1000.0);
    }

    stop() {
        window.clearInterval(this._instance);

        this._instance = null;
    }
}
