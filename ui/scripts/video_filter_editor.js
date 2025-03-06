import {Button} from "../scripts/base/button.js";
import {Checkbox} from "../scripts/base/checkbox.js";
import {Column} from "../scripts/base/column.js";
import {ReorderableAccordion} from "../scripts/base/reorderable_list.js";
import {FieldManager} from "../scripts/core/field_manager.js";
import {Signal} from "../scripts/core/signal.js";
import {createElement} from "../scripts/utils/dom.js";
import {ConfigurableParamForm} from "../scripts/configurable_param_form.js";

export class VideoFilterEditor extends ReorderableAccordion {
    constructor(definition) {
        super();

        this.onValueChange = new Signal();
        this.onRemove = new Signal();

        this._manager = new FieldManager(this.onValueChange);
        this._manager.value = {__type__: definition.type, enabled: true};

        this._header.insertBefore(createElement(null, Checkbox, (e) => {
            e.value = true;
            this._manager.manage(e, "enabled");
        }), this._header.firstChild.nextSibling);

        this.createChild(Column, (e) => {
            e.createChild(ConfigurableParamForm, null, definition.parameters, this._manager);

            e.createChild(Button, (e) => {
                e.label = "\u{f2ed} Remove";
                e.onClick.connect(() => {
                    this.parentElement.removeChild(this);

                    this.onRemove.fire();
                });
            });
        });
    }

    get value() {
        return this._manager.value;
    }

    set value(value) {
        this._manager.value = value;
    }
}
customElements.define("video-filter-editor", VideoFilterEditor);
