import {Signal} from "../../scripts/core/signal.js";

export class DragController {
    constructor() {
        this.onStart = new Signal();
        this.onMove = new Signal();
        this.onEnd = new Signal();

        this._enabled = true;
        this._element = null;

        window.addEventListener("touchmove", (event) => {
            if (!this._enabled || !this._element) return;

            event.stopPropagation();
            event.preventDefault();
        }, {passive: false});

        window.addEventListener("pointermove", (event) => {
            if (!this._enabled || !this._element) return;

            event.stopPropagation();

            this.onMove.fire(this._element, event);
        });

        window.addEventListener("pointerup", (event) => {
            if (!this._enabled || !this._element) return;

            event.stopPropagation();

            this.onEnd.fire(this._element);

            this._element = null;
        });
    }

    get enabled() {
        return this._enabled;
    }

    set enabled(value) {
        this._enabled = value;

        if (!value && this._element) {
            this.onEnd.fire(this._element);

            this._element = null;
        }
    }

    register(element) {
        element.addEventListener("pointerdown", (event) => {
            event.stopPropagation();

            this.onStart.fire(element);

            this._element = element;
        });
    }
}
