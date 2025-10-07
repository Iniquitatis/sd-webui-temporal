import {Button} from "/scripts/base/button.js";
import {Column} from "/scripts/base/column.js";
import {Dropdown} from "/scripts/base/dropdown.js";
import {Form} from "/scripts/base/form.js";
import {ChoiceListEditor, ListEditor} from "/scripts/base/list_editor.js";
import {NumberBox} from "/scripts/base/number_box.js";
import {ReorderableAccordion, ReorderableElement} from "/scripts/base/reorderable_list.js";
import {Row} from "/scripts/base/row.js";
import {ToolButton} from "/scripts/base/tool_button.js";
import {FieldManager} from "/scripts/core/field_manager.js";
import {Signal} from "/scripts/core/signal.js";
import {deepCopy, mapValues} from "/scripts/utils/object.js";
import {getFieldDefinition} from "/scripts/object_field.js";

class KeyframeEditor extends ReorderableElement {
    constructor(schema) {
        super();

        this.onValueChange = new Signal();
        this.onDuplicateRequest = new Signal();
        this.onRemoveRequest = new Signal();

        this._manager = new FieldManager(this.onValueChange);

        this.createChild(NumberBox, (e) => {
            e.style.width = "100%";
            e.minimum = 1;
            e.step = 1;
            e.value = 1;
            this._manager.manage(e, "frame");
        });

        let {cls, initializer, args, reader, writer} = getFieldDefinition(schema);

        this.createChild(cls, (e) => {
            initializer(e);
            e.style.width = "100%";
            this._manager.manage(e, "value", reader, writer);
        }, ...(args ?? []));

        this.createChild(ToolButton, (e) => {
            e.label = "\u{f0c5}";
            e.onClick.connect(() => {
                this.onDuplicateRequest.fire();
            });
        });

        this.createChild(ToolButton, (e) => {
            e.label = "\u{f2ed}";
            e.onClick.connect(() => {
                this.onRemoveRequest.fire();
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
customElements.define("ce-keyframe-editor", KeyframeEditor);

class KeyframeList extends ListEditor {
    constructor(schema) {
        super(KeyframeEditor);

        this._schema = schema;
    }

    getArgsFromItem(item) {
        return [this._schema];
    }

    getDefaultArgs() {
        return [this._schema];
    }
}
customElements.define("ce-keyframe-list", KeyframeList);

class TrackEditor extends ReorderableAccordion {
    constructor(key, schema) {
        super();

        this.onValueChange = new Signal();
        this.onDuplicateRequest = new Signal();
        this.onRemoveRequest = new Signal();

        this._manager = new FieldManager(this.onValueChange);
        this._manager.value.key = key;

        this.label = schema.name;

        this.createChild(Form, (e) => {
            e.createRow((e) => {
                e.createField("Interpolation", Dropdown, (e) => {
                    e.style.width = "100%";
                    e.choices = {
                        "linear": "Linear",
                        "smoothstep": "Smoothstep",
                        "smootherstep": "Smootherstep",
                        "step": "Step",
                        "step_start": "Step start",
                        "step_end": "Step end",
                    };
                    this._manager.manage(e, "interpolation")
                });

                e.createField("Bounds", Dropdown, (e) => {
                    e.style.width = "100%";
                    e.choices = {
                        "clamp": "Clamp",
                        "repeat": "Repeat",
                        "mirror": "Mirror",
                    };
                    this._manager.manage(e, "bounds");
                });
            });

            e.createChild(KeyframeList, (e) => {
                e.addLabel = "\u{e59e} Add keyframe";
                this._manager.manage(e, "keyframes");
            }, schema);

            e.createChild(Row, (e) => {
                e.createChild(Button, (e) => {
                    e.label = "\u{f0c5} Duplicate";
                    e.style.width = "100%";
                    e.onClick.connect(() => {
                        this.onDuplicateRequest.fire(deepCopy(this.value));
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
customElements.define("ce-track-editor", TrackEditor);

class TrackList extends ChoiceListEditor {
    constructor(schema) {
        super(TrackEditor, mapValues(schema.fields, (_, schema) => schema.name));

        this._schema = schema;
    }

    getArgsFromItem(item) {
        return [item.key, this._schema.fields[item.key]];
    }

    getArgsFromChoice(choice) {
        return [choice, this._schema.fields[choice]];
    }
}
customElements.define("ce-track-list", TrackList);

export class AnimationEditor extends Column {
    constructor(schema) {
        super();

        this.onValueChange = new Signal();

        this._manager = new FieldManager(this.onValueChange);

        this.createChild(TrackList, (e) => {
            e.addLabel = "\u{e59e} Add track";
            this._manager.manage(e, "tracks");
        }, schema);
    }

    get value() {
        return this._manager.value;
    }

    set value(value) {
        this._manager.value = value;
    }
}
customElements.define("ce-animation-editor", AnimationEditor);
