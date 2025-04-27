import {Button} from "../scripts/base/button.js";
import {Checkbox} from "../scripts/base/checkbox.js";
import {Column} from "../scripts/base/column.js";
import {ReorderableAccordion} from "../scripts/base/reorderable_list.js";
import {Row} from "../scripts/base/row.js";
import {FieldManager} from "../scripts/core/field_manager.js";
import {Signal} from "../scripts/core/signal.js";
import {deepCopy} from "../scripts/utils/object.js";
import {ObjectForm} from "../scripts/object_form.js";
import {objectTypes} from "../scripts/shared_data.js";

export class VideoFilterEditor extends ReorderableAccordion {
    constructor(type) {
        super();

        this.onValueChange = new Signal();
        this.onRemove = new Signal();

        this._manager = new FieldManager(this.onValueChange);
        this._manager.value.__type__ = type;

        let schema = objectTypes[type];

        this.createBeforeLabel(Checkbox, (e) => {
            e.value = schema.fields.enabled.default;
            this._manager.manage(e, "enabled");
        });

        this.createChild(Column, (e) => {
            let filteredKeys = [];

            for (let key of Object.keys(schema.fields)) {
                if (key != "enabled") {
                    filteredKeys.push(key);
                }
            }

            if (filteredKeys.length > 0) {
                e.createChild(ObjectForm, (e) => {
                    e.manageMultiple(filteredKeys);
                }, schema.type, this._manager);
            } else {
                e.createChild("span", (e) => {
                    e.innerText = "This filter has no configurable paremeters.";
                });
            }

            e.createChild(Row, (e) => {
                e.createChild(Button, (e) => {
                    e.label = "\u{f0c5} Duplicate";
                    e.style.width = "100%";
                    e.onClick.connect(() => {
                        let newValue = deepCopy(this.value);
                        delete newValue.__id__;
                        // FIXME: Hacky. Should somehow interact with the
                        // ModuleList (which is one level higher), not
                        // ReorderableList.
                        this.parentElement.parentElement._createModule(type, newValue);
                    });
                });

                e.createChild(Button, (e) => {
                    e.label = "\u{f2ed} Remove";
                    e.style.width = "100%";
                    e.onClick.connect(() => {
                        this.parentElement.removeChild(this);

                        this.onRemove.fire();
                    });
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
