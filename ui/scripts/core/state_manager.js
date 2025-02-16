import {Signal} from "../../scripts/core/signal.js";

export class StateManager {
    constructor() {
        this.onStatesChange = new Signal();
        this.onValueChange = new Signal();

        this._states = {};
        this._values = [];
        this._data = [];
        this._index = 0;
    }

    get data() {
        return this._data[this._index];
    }

    get index() {
        return this._index;
    }

    get states() {
        return this._states;
    }

    get value() {
        return this._values[this._index];
    }

    set index(value) {
        this._index = value;

        this.onValueChange.fire(this.value, this.data);
    }

    set states(value) {
        this._states = value;
        this._values = Object.keys(value);
        this._data = Object.values(value);
        this._index = Math.min(Math.max(this._index, 0), this._values.length);

        this.onStatesChange.fire(value, this._values, this._data);
    }

    set value(value) {
        this._index = this._values.indexOf(value);

        this.onValueChange.fire(this.value, this.data);
    }

    nextState() {
        this._index++;
        this._index %= this._values.length;

        this.onValueChange.fire(this.value, this.data);
    }
}
