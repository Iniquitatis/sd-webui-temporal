import {Button} from "../scripts/base/button.js";
import {Column} from "../scripts/base/column.js";
import {ReorderableAccordion} from "../scripts/base/reorderable_list.js";
import {FieldManager} from "../scripts/core/field_manager.js";
import {Signal} from "../scripts/core/signal.js";
import {createElement} from "../scripts/utils/dom.js";
import {ConfigurableParamEditor} from "../scripts/configurable_param_editor.js";

export class VideoFilterEditor extends ReorderableAccordion {
    constructor(definition) {
        super();

        this.onValueChange = new Signal();
        this.onRemove = new Signal();

        this._manager = new FieldManager(this.onValueChange);
        this._manager.value = {id: definition.id, enabled: true};

        this._header.insertBefore(createElement(null, "input", (e) => {
            e.type = "checkbox";
            e.checked = true;
            e.addEventListener("change", () => {
                this._manager._value.enabled = e.checked;

                this.onValueChange.fire(this._value);
            });
            this._manager._onValueReceive.connect((value) => {
                e.checked = value.enabled;
            });
        }), this._header.firstChild.nextSibling);

        this.createChild(Column, (e) => {
            for (let [id, param] of Object.entries(definition.parameters)) {
                e.createChild(ConfigurableParamEditor, (e) => {
                    this._manager.manage(e, id);
                }, param);
            }

            e.createChild(Button, (e) => {
                e.label = "\u{274c}\u{fe0e} Remove";
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
