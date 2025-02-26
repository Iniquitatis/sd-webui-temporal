import {Button} from "../scripts/base/button.js";
import {Checkbox} from "../scripts/base/checkbox.js";
import {Form} from "../scripts/base/form.js";
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
        this._manager.value = {__type__: definition.type, enabled: true};

        this._header.insertBefore(createElement(null, Checkbox, (e) => {
            e.value = true;
            this._manager.manage(e, "enabled");
        }), this._header.firstChild.nextSibling);

        this.createChild(Form, (e) => {
            for (let [id, param] of Object.entries(definition.parameters)) {
                e.createField(param.name, ConfigurableParamEditor, (e) => {
                    this._manager.manage(e, id);
                }, param);
            }

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
