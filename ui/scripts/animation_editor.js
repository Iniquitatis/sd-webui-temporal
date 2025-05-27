import {Button} from "../scripts/base/button.js";
import {Checkbox} from "../scripts/base/checkbox.js";
import {ColorPicker} from "../scripts/base/color_picker.js";
import {Column} from "../scripts/base/column.js";
import {Dropdown} from "../scripts/base/dropdown.js";
import {Form} from "../scripts/base/form.js";
import {ChoiceListEditor, ListEditor} from "../scripts/base/list_editor.js";
import {NumberBox} from "../scripts/base/number_box.js";
import {ReorderableAccordion, ReorderableElement} from "../scripts/base/reorderable_list.js";
import {Row} from "../scripts/base/row.js";
import {TextBox} from "../scripts/base/text_box.js";
import {ToolButton} from "../scripts/base/tool_button.js";
import {FieldManager} from "../scripts/core/field_manager.js";
import {Signal} from "../scripts/core/signal.js";
import {deepCopy, mapValues} from "../scripts/utils/object.js";

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

        if (schema.type == "bool") {
            this.createChild(Checkbox, (e) => {
                e.style.width = "100%";
                e.value = schema.default ? deepCopy(schema.default) : false;
                this._manager.manage(e, "value");
            });
        } else if (schema.type == "int") {
            this.createChild(NumberBox, (e) => {
                e.style.width = "100%";
                e.minimum = schema.minimum ?? undefined;
                e.maximum = schema.maximum ?? undefined;
                e.step = schema.step ?? 1;
                e.value = schema.default ?? schema.minimum;
                this._manager.manage(e, "value");
            });
        } else if (schema.type == "float") {
            // FIXME: Treated as an integer by the backend when the value is
            // rounded
            this.createChild(NumberBox, (e) => {
                e.style.width = "100%";
                e.minimum = schema.minimum ?? undefined;
                e.maximum = schema.maximum ?? undefined;
                e.step = schema.step ?? 0.1;
                e.value = schema.default ?? schema.minimum;
                this._manager.manage(e, "value");
            });
        } else if (schema.type == "str") {
            this.createChild(TextBox, (e) => {
                e.style.width = "100%";
                e.value = schema.default ?? "";
                this._manager.manage(e, "value");
            });
        } else if (schema.type == "temporal.color.Color") {
            this.createChild(ColorPicker, (e) => {
                e.style.width = "100%";
                e.value = deepCopy(schema.default) ?? {r: 0.0, g: 0.0, b: 0.0, a: 1.0};
                this._manager.manage(e, "value");
            }, schema.channels ?? 3);
        } else {
            this.createChild("span", (e) => {
                e.innerText = "Unhandled";
            });
        }

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
customElements.define("animation-keyframe-editor", KeyframeEditor);

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
customElements.define("animation-keyframe-list", KeyframeList);

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
                        let newValue = deepCopy(this.value);
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
customElements.define("animation-track-editor", TrackEditor);

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
customElements.define("animation-track-list", TrackList);

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
customElements.define("animation-editor", AnimationEditor);
