import {Button} from "../scripts/base/button.js";
import {Checkbox} from "../scripts/base/checkbox.js";
import {Column} from "../scripts/base/column.js";
import {ChoiceListEditor} from "../scripts/base/list_editor.js";
import {ReorderableAccordion} from "../scripts/base/reorderable_list.js";
import {Row} from "../scripts/base/row.js";
import {FieldManager} from "../scripts/core/field_manager.js";
import {Signal} from "../scripts/core/signal.js";
import {deepCopy, mapValues} from "../scripts/utils/object.js";
import {ObjectForm} from "../scripts/object_form.js";
import {videoFilters} from "../scripts/shared_data.js";

class VideoFilterEditor extends ReorderableAccordion {
    constructor(type) {
        super();

        this.onValueChange = new Signal();
        this.onDuplicateRequest = new Signal();
        this.onRemoveRequest = new Signal();

        this._manager = new FieldManager(this.onValueChange);
        this._manager.value.__type__ = type;

        let schema = videoFilters[type];

        this.label = schema.name;

        this.createBeforeLabel(Checkbox, (e) => {
            e.value = schema.fields.enabled.default;
            this._manager.manage(e, "enabled");
        });

        this.createChild(Column, (e) => {
            let filteredKeys = Object.keys(schema.fields).filter((key) => {
                return key != "enabled";
            });

            if (filteredKeys.length > 0) {
                e.createChild(ObjectForm, (e) => {
                    e.manageMultiple(filteredKeys);
                }, schema.type, this._manager);
            } else {
                e.createChild("span", (e) => {
                    e.innerText = "This filter has no configurable parameters.";
                });
            }

            e.createChild(Row, (e) => {
                e.createChild(Button, (e) => {
                    e.label = "\u{f0c5} Duplicate";
                    e.style.width = "100%";
                    e.onClick.connect(() => {
                        let newValue = deepCopy(this.value);
                        // FIXME: Doesn't "deep drop", though
                        delete newValue.__id__;
                        this.onDuplicateRequest.fire(newValue);
                    });
                });

                e.createChild(Button, (e) => {
                    e.label = "\u{f2ed} Remove";
                    e.style.width = "100%";
                    e.onClick.connect(() => {
                        this.onRemoveRequest.fire();
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

export class VideoFilterList extends ChoiceListEditor {
    constructor() {
        super(VideoFilterEditor, mapValues(videoFilters, (_, schema) => schema.name));
    }

    getArgsFromItem(item) {
        return [item.__type__];
    }

    getArgsFromChoice(choice) {
        return [choice];
    }
}
customElements.define("video-filter-list", VideoFilterList);
