import {Block} from "../scripts/base/block.js";
import {CodeArea} from "../scripts/base/code_area.js";
import {FieldManager} from "../scripts/core/field_manager.js";
import {Signal} from "../scripts/core/signal.js";

export class AnimationEditor extends Block {
    constructor() {
        super();

        this.onValueChange = new Signal();

        this._manager = new FieldManager(this.onValueChange);

        this._box = this.createChild(CodeArea, (e) => {
            this._manager.manage(e, "code");
        });
    }

    get value() {
        return this._manager.value;
    }

    set value(value) {
        this._manager.value = value;
    }
}
customElements.define("animation-editor", AnimationEditor);
