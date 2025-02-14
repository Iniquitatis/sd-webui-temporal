import {ValueEditor} from "../../scripts/base/value_editor.js";
import {FieldManager} from "../../scripts/core/field_manager.js";
import {Signal} from "../../scripts/core/signal.js";

export class VectorEditor extends ValueEditor {
    constructor(cls, axes) {
        super();

        this.onValueChange = new Signal();

        this._manager = new FieldManager(this.onValueChange);
        this._editors = [];

        for (let [key, label] of Object.entries(axes)) {
            this._editors.push(this._content.createChild(cls, (e) => {
                e.label = label;
                this._manager.manage(e, key);
            }));
        }
    }

    get maximum() {
        return this._editors[0].maximum;
    }

    get minimum() {
        return this._editors[0].minimum;
    }

    get step() {
        return this._editors[0].step;
    }

    get value() {
        return this._manager.value;
    }

    set maximum(value) {
        for (let slider of this._editors) {
            slider.maximum = value;
        }
    }

    set minimum(value) {
        for (let slider of this._editors) {
            slider.minimum = value;
        }
    }

    set step(value) {
        for (let slider of this._editors) {
            slider.step = value;
        }
    }

    set value(value) {
        this._manager.value = value;
    }
}
customElements.define("vector-editor", VectorEditor);
