import {Signal} from "/scripts/core/signal.js";

export class FieldManager {
    constructor(signal = null) {
        this.onFieldChange = new Signal();
        this.onValueChange = signal ?? new Signal();
        this._onValueReceive = new Signal();
        this._value = {};
    }

    get value() {
        return this._value;
    }

    set value(value) {
        this._value = value;

        this.onValueChange.withDisabled(() => {
            this._onValueReceive.fire(value);
        });
        this.onValueChange.fire(this.value);
    }

    manage(widget, key, reader = null, writer = null) {
        this._value[key] = writer ? writer(widget.value) : widget.value;

        this._onValueReceive.connect((value) => {
            if (value.hasOwnProperty(key)) {
                widget.value = reader ? reader(value[key]) : value[key];
            } else {
                console.log(`${value} doesn't contain ${key} field`);
            }
        });

        widget.onValueChange.connect((value) => {
            this._value[key] = writer ? writer(value) : value;

            this.onFieldChange.fire(key, this._value[key]);
            this.onValueChange.fire(this.value);
        });
    }
}
