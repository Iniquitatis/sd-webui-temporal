import {Button} from "../../scripts/base/button.js";
import {Signal} from "../../scripts/core/signal.js";

export class MultiStateButton extends Button {
    constructor(states) {
        super();

        this._states = Object.keys(states);
        this._labels = Object.values(states);
        this._index = 0;

        this.addEventListener("click", () => {
            this._index++;
            this._index %= this._states.length;
            this._button.innerText = this._labels[this._index];

            this.onStateChange.fire(this.state);
        });

        this._button.innerText = this._labels[0];

        this.onStateChange = new Signal();
    }

    get state() {
        return this._states[this._index];
    }

    set state(value) {
        this._index = this._states.indexOf(value);
        this._button.innerText = this._labels[this._index];

        this.onStateChange.fire(this.state);
    }
}
customElements.define("multi-state-button", MultiStateButton);
