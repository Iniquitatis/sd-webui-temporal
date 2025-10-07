import {Checkbox} from "/scripts/base/checkbox.js";
import {CodeArea} from "/scripts/base/code_area.js";
import {ColorPicker} from "/scripts/base/color_picker.js";
import {Dropdown} from "/scripts/base/dropdown.js";
import {ImageBox} from "/scripts/base/image_box.js";
import {NumberBox} from "/scripts/base/number_box.js";
import {Radio} from "/scripts/base/radio.js";
import {Slider} from "/scripts/base/slider.js";
import {TextArea} from "/scripts/base/text_area.js";
import {TextBox} from "/scripts/base/text_box.js";
import {VectorEditor} from "/scripts/base/vector_editor.js";
import {VideoBox} from "/scripts/base/video_box.js";
import {deepCopy} from "/scripts/utils/object.js";
import {AnimationEditor} from "/scripts/animation_editor.js";
import {PipelineModuleList} from "/scripts/pipeline_module_list.js";
import {SeedBox} from "/scripts/seed_box.js";
import {blendModes} from "/scripts/shared_data.js";
import {VideoFilterList} from "/scripts/video_filter_list.js";

export function getFieldDefinition(schema) {
    let typeParts = [schema.type];

    if (schema.subtype) {
        typeParts.push(`[${schema.subtype}]`);
    }

    if (schema.display) {
        typeParts.push(`<${schema.display}>`);
    }

    let fullType = typeParts.join("");

    if (!EDITORS.hasOwnProperty(fullType)) {
        console.log(`Unhandled type ${fullType}`);
        return;
    }

    let {cls, args, initializer, reader, writer} = EDITORS[fullType](schema);

    return {cls, args, initializer, reader, writer};
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

    "modules.animation.Animation": (schema) => ({
        cls: AnimationEditor,
        initializer: (e) => {
            e.value = deepCopy(schema.default) ?? {};
        },
    }),

    "modules.blend_modes.BlendMode": (schema) => ({
        cls: Dropdown,
        initializer: (e) => {
            e.choices = blendModes;
            e.value = schema.default.__type__ ?? "modules.blend_modes.NormalBlendMode";
        },
        reader: (value) => value.__type__,
        writer: (value) => ({__type__: value}),
    }),

    "modules.color.Color": (schema) => ({
        cls: ColorPicker,
        args: [schema.channels ?? 3],
        initializer: (e) => {
            e.value = deepCopy(schema.default) ?? {r: 0.0, g: 0.0, b: 0.0, a: 1.0};
        },
    }),

    "modules.seed.Seed": (schema) => ({
        cls: SeedBox,
        initializer: (e) => {
            e.value = schema.default ?? -1;
        },
    }),

    "modules.utils.image.NumpyImage": (schema) => ({
        cls: ImageBox,
        args: [["clear", "download", "edit", "fullscreen", "upload"]],
        initializer: (e) => {
            e.style.height = "12rem";
            e.channels = schema.channels ?? 3;
            e.value = schema.default ?? null;
        },
    }),

    "modules.vector.IntVector<box>": (schema) => ({
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

    "modules.vector.IntVector<slider>": (schema) => ({
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

    "modules.vector.FloatVector<box>": (schema) => ({
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

    "modules.vector.FloatVector<slider>": (schema) => ({
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

    "modules.video.Video": (schema) => ({
        cls: VideoBox,
        args: [["clear", "download", "fullscreen", "upload"]],
        initializer: (e) => {
            e.style.height = "12rem";
            e.value = schema.default ?? null;
        },
    }),

    "list[modules.pipeline_module.PipelineModule]": (schema) => ({
        cls: PipelineModuleList,
        args: [],
        initializer: (e) => {
            e.value = deepCopy(schema.default) ?? [];
        },
    }),

    "list[modules.video_filter.VideoFilter]": (schema) => ({
        cls: VideoFilterList,
        args: [],
        initializer: (e) => {
            e.value = deepCopy(schema.default) ?? [];
        },
    }),
};
