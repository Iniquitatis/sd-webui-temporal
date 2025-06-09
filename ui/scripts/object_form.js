import {Accordion} from "../scripts/base/accordion.js";
import {Checkbox} from "../scripts/base/checkbox.js";
import {CodeArea} from "../scripts/base/code_area.js";
import {ColorPicker} from "../scripts/base/color_picker.js";
import {Dropdown} from "../scripts/base/dropdown.js";
import {Form} from "../scripts/base/form.js";
import {GroupBox} from "../scripts/base/group_box.js";
import {ImageBox} from "../scripts/base/image_box.js";
import {NumberBox} from "../scripts/base/number_box.js";
import {Radio} from "../scripts/base/radio.js";
import {Slider} from "../scripts/base/slider.js";
import {Tabs} from "../scripts/base/tabs.js";
import {TextArea} from "../scripts/base/text_area.js";
import {TextBox} from "../scripts/base/text_box.js";
import {VectorEditor} from "../scripts/base/vector_editor.js";
import {VideoBox} from "../scripts/base/video_box.js";
import {FieldManager} from "../scripts/core/field_manager.js";
import {Signal} from "../scripts/core/signal.js";
import {deepCopy} from "../scripts/utils/object.js";
import {AnimationEditor} from "../scripts/animation_editor.js";
import {PipelineModuleList} from "../scripts/pipeline_module_list.js";
import {SeedBox} from "../scripts/seed_box.js";
import {blendModes, objectTypes} from "../scripts/shared_data.js";
import {VideoFilterList} from "../scripts/video_filter_list.js";

export class ObjectForm extends Form {
    constructor(type, manager = null) {
        super(false);

        this.onValueChange = manager ? manager.onValueChange : new Signal();

        this._type = type;
        this._manager = manager ?? new FieldManager(this.onValueChange);
        if (!manager) this._manager.value.__type__ = type;
        this._lastTabs = null;
    }

    manage(key) {
        let field = objectTypes[this._type].fields[key];

        if (field.display == "accordion") {
            this._lastTabs = null;

            this.createChild(Accordion, (e) => {
                e.label = field.name;

                e.createChild(ObjectForm, (e) => {
                    e.manageAll();
                    e.value = deepCopy(field.default) ?? {};
                    this._manager.manage(e, key);
                }, field.type);
            });
        } else if (field.display == "group") {
            this._lastTabs = null;

            this.createChild(GroupBox, (e) => {
                e.label = field.name;

                e.createChild(ObjectForm, (e) => {
                    e.manageAll();
                    e.value = deepCopy(field.default) ?? {};
                    this._manager.manage(e, key);
                }, field.type);
            });
        } else if (field.display == "tab") {
            if (!this._lastTabs) {
                this._lastTabs = this.createChild(Tabs);
            }

            this._lastTabs.createTab(field.name, ObjectForm, (e) => {
                e.manageAll();
                e.value = deepCopy(field.default) ?? {};
                this._manager.manage(e, key);
            }, field.type);
        } else {
            this._lastTabs = null;

            let typeParts = [field.type];

            if (field.subtype) {
                typeParts.push(`[${field.subtype}]`);
            }

            if (field.display) {
                typeParts.push(`<${field.display}>`);
            }

            let fullType = typeParts.join("");

            if (!EDITORS.hasOwnProperty(fullType)) {
                console.log(`Unhandled type ${fullType}`);
                return;
            }

            let {cls, args, initializer, reader, writer} = EDITORS[fullType](field);

            this.createField(field.name, cls, (e) => {
                initializer(e);
                this._manager.manage(e, key, reader, writer);

                if (field.dependencies) {
                    this.onValueChange.connect((value) => {
                        e.formItem.visible = areDependenciesSatisfied(value, field.dependencies);
                    });
                }
            }, ...(args ?? []));
        }
    }

    manageMultiple(keys) {
        for (let key of keys) {
            this.manage(key);
        }
    }

    manageAll() {
        for (let key of Object.keys(objectTypes[this._type].fields)) {
            this.manage(key);
        }
    }

    get value() {
        return this._manager.value;
    }

    set value(value) {
        this._manager.value = value;
    }
}
customElements.define("object-form", ObjectForm);

function areDependenciesSatisfied(formValue, dependencies) {
    for (let [depKey, depValue] of Object.entries(dependencies)) {
        if (depValue instanceof Array) {
            if (!depValue.some((depChild) => formValue[depKey] == depChild)) {
                return false;
            }
        } else {
            if (formValue[depKey] != depValue) {
                return false;
            }
        }
    }

    return true;
}

const EDITORS = {
    "bool": (schema) => ({
        cls: Checkbox,
        initializer: (e) => {
            e.value = schema.default ?? false;
        },
    }),

    "int<box>": (schema) => ({
        cls: NumberBox,
        initializer: (e) => {
            e.minimum = schema.minimum ?? undefined;
            e.maximum = schema.maximum ?? undefined;
            e.step = schema.step ?? 1;
            e.value = schema.default ?? e.minimum;
        },
    }),

    "int<slider>": (schema) => ({
        cls: Slider,
        initializer: (e) => {
            e.minimum = schema.minimum ?? undefined;
            e.maximum = schema.maximum ?? undefined;
            e.step = schema.step ?? 1;
            e.suffix = schema.suffix ?? "";
            e.value = schema.default ?? e.minimum;
        },
    }),

    "float<box>": (schema) => ({
        cls: NumberBox,
        initializer: (e) => {
            e.minimum = schema.minimum ?? undefined;
            e.maximum = schema.maximum ?? undefined;
            e.step = schema.step ?? 0.1;
            e.value = schema.default ?? e.minimum;
        },
    }),

    "float<slider>": (schema) => ({
        cls: Slider,
        initializer: (e) => {
            e.minimum = schema.minimum ?? undefined;
            e.maximum = schema.maximum ?? undefined;
            e.step = schema.step ?? 0.1;
            e.suffix = schema.suffix ?? "";
            e.value = schema.default ?? e.minimum;
        },
    }),

    "str<area>": (schema) => ({
        cls: TextArea,
        initializer: (e) => {
            e.value = schema.default ?? "";
        },
    }),

    "str<box>": (schema) => ({
        cls: TextBox,
        initializer: (e) => {
            e.value = schema.default ?? "";
        },
    }),

    "str<code>": (schema) => ({
        cls: CodeArea,
        initializer: (e) => {
            e.value = schema.default ?? "";
        },
    }),

    "str<menu>": (schema) => ({
        cls: Dropdown,
        initializer: (e) => {
            e.choices = schema.choices ?? {"": ""};
            e.value = schema.default ?? null;
        },
    }),

    "str<radio>": (schema) => ({
        cls: Radio,
        initializer: (e) => {
            e.choices = schema.choices ?? {"": ""};
            e.value = schema.default ?? null;
        },
    }),

    "pathlib.Path": (schema) => ({
        cls: TextBox,
        initializer: (e) => {
            e.value = schema.default ?? "";
        },
    }),

    "temporal.animation.Animation": (schema) => ({
        cls: AnimationEditor,
        initializer: (e) => {
            e.value = deepCopy(schema.default) ?? {};
        },
    }),

    "temporal.blend_modes.BlendMode": (schema) => ({
        cls: Dropdown,
        initializer: (e) => {
            e.choices = blendModes;
            e.value = schema.default.__type__ ?? "temporal.blend_modes.NormalBlendMode";
        },
        reader: (value) => value.__type__,
        writer: (value) => ({__type__: value}),
    }),

    "temporal.color.Color": (schema) => ({
        cls: ColorPicker,
        args: [schema.channels ?? 3],
        initializer: (e) => {
            e.value = deepCopy(schema.default) ?? {r: 0.0, g: 0.0, b: 0.0, a: 1.0};
        },
    }),

    "temporal.seed.Seed": (schema) => ({
        cls: SeedBox,
        initializer: (e) => {
            e.value = schema.default ?? -1;
        },
    }),

    "temporal.utils.image.NumpyImage": (schema) => ({
        cls: ImageBox,
        args: [["clear", "download", "edit", "fullscreen", "upload"]],
        initializer: (e) => {
            e.style.height = "12rem";
            e.channels = schema.channels ?? 3;
            e.value = schema.default ?? null;
        },
    }),

    "temporal.vector.IntVector<box>": (schema) => ({
        cls: VectorEditor,
        args: [NumberBox, {
            x: schema.axes?.[0] ?? "X",
            y: schema.axes?.[1] ?? "Y",
        }],
        initializer: (e) => {
            e.minimum = schema.minimum ?? undefined;
            e.maximum = schema.maximum ?? undefined;
            e.step = schema.step ?? 1;
            e.value = deepCopy(schema.default) ?? {x: e.minimum, y: e.minimum};
        },
    }),

    "temporal.vector.IntVector<slider>": (schema) => ({
        cls: VectorEditor,
        args: [Slider, {
            x: schema.axes?.[0] ?? "X",
            y: schema.axes?.[1] ?? "Y",
        }],
        initializer: (e) => {
            e.minimum = schema.minimum ?? undefined;
            e.maximum = schema.maximum ?? undefined;
            e.step = schema.step ?? 1;
            e.suffix = schema.suffix ?? "";
            e.value = deepCopy(schema.default) ?? {x: e.minimum, y: e.minimum};
        },
    }),

    "temporal.vector.FloatVector<box>": (schema) => ({
        cls: VectorEditor,
        args: [NumberBox, {
            x: schema.axes?.[0] ?? "X",
            y: schema.axes?.[1] ?? "Y",
        }],
        initializer: (e) => {
            e.minimum = schema.minimum ?? undefined;
            e.maximum = schema.maximum ?? undefined;
            e.step = schema.step ?? 0.1;
            e.value = deepCopy(schema.default) ?? {x: e.minimum, y: e.minimum};
        },
    }),

    "temporal.vector.FloatVector<slider>": (schema) => ({
        cls: VectorEditor,
        args: [Slider, {
            x: schema.axes?.[0] ?? "X",
            y: schema.axes?.[1] ?? "Y",
        }],
        initializer: (e) => {
            e.minimum = schema.minimum ?? undefined;
            e.maximum = schema.maximum ?? undefined;
            e.step = schema.step ?? 0.1;
            e.suffix = schema.suffix ?? "";
            e.value = deepCopy(schema.default) ?? {x: e.minimum, y: e.minimum};
        },
    }),

    "temporal.video.Video": (schema) => ({
        cls: VideoBox,
        args: [["clear", "download", "fullscreen", "upload"]],
        initializer: (e) => {
            e.style.height = "12rem";
            e.value = schema.default ?? null;
        },
    }),

    "list[temporal.pipeline_module.PipelineModule]": (schema) => ({
        cls: PipelineModuleList,
        args: [],
        initializer: (e) => {
            e.value = deepCopy(schema.default) ?? [];
        },
    }),

    "list[temporal.video_filter.VideoFilter]": (schema) => ({
        cls: VideoFilterList,
        args: [],
        initializer: (e) => {
            e.value = deepCopy(schema.default) ?? [];
        },
    }),
};
