import {Signal} from "../../scripts/core/signal.js";

export class FieldManager {
    constructor(signal, reader = null, writer = null) {
        this._onValueReceive = new Signal();
        this._onValueChange = signal;
        this._reader = reader;
        this._writer = writer;
        this._value = {};
    }

    get value() {
        return this._reader ? this._reader(this._value) : this._value;
    }

    set value(value) {
        this._value = this._writer ? this._writer(value) : value;

        this._onValueChange.withDisabled(() => {
            this._onValueReceive.fire(this._value);
        });
        this._onValueChange.fire(this.value);
    }

    manage(widget, field, reader = null, writer = null) {
        this._onValueReceive.connect((value) => {
            if (value.hasOwnProperty(field)) {
                widget.value = reader ? reader(value[field]) : value[field];
            } else {
                console.log(`${value} doesn't contain ${field} field`);
            }
        });

        widget.onValueChange.connect((value) => {
            this._value[field] = writer ? writer(value) : value;

            this._onValueChange.fire(this.value);
        });
    }
}
