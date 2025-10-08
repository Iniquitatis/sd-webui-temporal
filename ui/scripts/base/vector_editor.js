import {Block} from "/scripts/base/block.js";
import {Row} from "/scripts/base/row.js";
import {FieldManager} from "/scripts/core/field_manager.js";
import {Signal} from "/scripts/core/signal.js";
import {defineElement} from "/scripts/utils/dom.js";

export class VectorEditor extends Row {
    static tag = "ce-vector-editor";
    static css = `
        <self> > .label {
            align-content: center;
            color: var(--hint-color);
            font-size: 0.9rem;
            height: var(--widget-height);
            width: 1rem;
        }

        <self> > .editor {
            width: 100%;
        }
    `;

    constructor(cls, axes) {
        super();

        this.onValueChange = new Signal();

        this._manager = new FieldManager(this.onValueChange);
        this._editors = [];

        for (let [key, label] of Object.entries(axes)) {
            this.createChild(Block, (e) => {
                e.className = "label";
                e.innerText = label;
            });

            this._editors.push(this.createChild(cls, (e) => {
                e.className = "editor";
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

    get suffix() {
        return this._editors[0].suffix;
    }

    get value() {
        return this._manager.value;
    }

    set maximum(value) {
        for (let editor of this._editors) {
            editor.maximum = value;
        }
    }

    set minimum(value) {
        for (let editor of this._editors) {
            editor.minimum = value;
        }
    }

    set step(value) {
        for (let editor of this._editors) {
            editor.step = value;
        }
    }

    set suffix(value) {
        for (let editor of this._editors) {
            editor.suffix = value;
        }
    }

    set value(value) {
        this._manager.value = value;
    }
}
defineElement(VectorEditor);
