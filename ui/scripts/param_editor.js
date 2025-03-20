import {Checkbox} from "../scripts/base/checkbox.js";
import {CodeArea} from "../scripts/base/code_area.js";
import {ColorPicker} from "../scripts/base/color_picker.js";
import {Dropdown} from "../scripts/base/dropdown.js";
import {ImageBox} from "../scripts/base/image_box.js";
import {NumberBox} from "../scripts/base/number_box.js";
import {Radio} from "../scripts/base/radio.js";
import {Slider} from "../scripts/base/slider.js";
import {TextArea} from "../scripts/base/text_area.js";
import {TextBox} from "../scripts/base/text_box.js";
import {VectorEditor} from "../scripts/base/vector_editor.js";
import {Signal} from "../scripts/core/signal.js";
import {Widget} from "../scripts/core/widget.js";
import {ImageSourceEditor} from "../scripts/image_source_editor.js";
import {SeedBox} from "../scripts/seed_box.js";
import {VideoRendererEditor} from "../scripts/video_renderer_editor.js";

export class ParamEditor extends Widget {
    constructor(definition) {
        super();

        this.onValueChange = new Signal();

        this._editor = null;

        let typeParts = [definition.type];

        if (definition.ui_type) {
            typeParts.push(definition.ui_type);
        }

        let fullType = typeParts.join("|");

        if (EDITORS.hasOwnProperty(fullType)) {
            this._editor = EDITORS[fullType](this, definition);
            this._editor.onValueChange.connect((value) => {
                this.onValueChange.fire(value);
            });
        } else {
            console.log(`WARNING: Unhandled type ${fullType}`);
        }
    }

    get value() {
        return this._editor.value;
    }

    set value(value) {
        this._editor.value = value;
    }

    attachTitle(title) {
        this._editor.attachTitle(title);
    }

    canAttachTitle() {
        return this._editor.canAttachTitle();
    }

    isComplexWidget() {
        return this._editor.isComplexWidget();
    }
}
customElements.define("param-editor", ParamEditor);

const EDITORS = {
    "bool": (parent, definition) => parent.createChild(Checkbox, (e) => {
        e.value = definition.default ?? false;
    }),

    "int|box": (parent, definition) => parent.createChild(NumberBox, (e) => {
        e.minimum = definition.minimum ?? undefined;
        e.maximum = definition.maximum ?? undefined;
        e.step = definition.step ?? 1;
        e.value = definition.default ?? e.minimum;
    }),

    "int|seed": (parent, definition) => parent.createChild(SeedBox, (e) => {
        e.value = definition.default ?? -1;
    }),

    "int|slider": (parent, definition) => parent.createChild(Slider, (e) => {
        e.minimum = definition.minimum ?? undefined;
        e.maximum = definition.maximum ?? undefined;
        e.step = definition.step ?? 1;
        e.value = definition.default ?? e.minimum;
    }),

    "float|box": (parent, definition) => parent.createChild(NumberBox, (e) => {
        e.minimum = definition.minimum ?? undefined;
        e.maximum = definition.maximum ?? undefined;
        e.step = definition.step ?? 0.1;
        e.value = definition.default ?? e.minimum;
    }),

    "float|slider": (parent, definition) => parent.createChild(Slider, (e) => {
        e.minimum = definition.minimum ?? undefined;
        e.maximum = definition.maximum ?? undefined;
        e.step = definition.step ?? 0.1;
        e.value = definition.default ?? e.minimum;
    }),

    "str|area": (parent, definition) => parent.createChild(TextArea, (e) => {
        e.value = definition.default ?? "";
    }),

    "str|box": (parent, definition) => parent.createChild(TextBox, (e) => {
        e.value = definition.default ?? "";
    }),

    "str|code": (parent, definition) => parent.createChild(CodeArea, (e) => {
        e.value = definition.default ?? "";
    }),

    "str|menu": (parent, definition) => parent.createChild(Dropdown, (e) => {
        e.choices = definition.choices ?? {"": ""};
        e.value = definition.default ?? null;
    }),

    "str|radio": (parent, definition) => parent.createChild(Radio, (e) => {
        e.choices = definition.choices ?? {"": ""};
        e.value = definition.default ?? null;
    }),

    "pathlib.Path": (parent, definition) => parent.createChild(TextBox, (e) => {
        e.value = definition.default ?? "";
    }),

    "temporal.color.Color": (parent, definition) => parent.createChild(ColorPicker, (e) => {
        e.value = definition.default ?? {r: 0.0, g: 0.0, b: 0.0, a: 1.0};
    }, definition.channels ?? 3),

    "temporal.image_source.ImageSource": (parent, definition) => parent.createChild(ImageSourceEditor, (e) => {
        e.channels = definition.channels ?? 3;
    }),

    "temporal.utils.image.NumpyImage": (parent, definition) => parent.createChild(ImageBox, (e) => {
        e.channels = definition.channels ?? 3;
    }, ["clear", "download", "fullscreen", "upload"]),

    "temporal.vector.IntVector|box": (parent, definition) => parent.createChild(VectorEditor, (e) => {
        e.minimum = definition.minimum ?? undefined;
        e.maximum = definition.maximum ?? undefined;
        e.step = definition.step ?? 1;
        e.value = definition.default ?? {x: e.minimum, y: e.minimum};
    }, NumberBox, {
        x: definition.axes?.[0] ?? "X",
        y: definition.axes?.[1] ?? "Y",
    }),

    "temporal.vector.IntVector|slider": (parent, definition) => parent.createChild(VectorEditor, (e) => {
        e.minimum = definition.minimum ?? undefined;
        e.maximum = definition.maximum ?? undefined;
        e.step = definition.step ?? 1;
        e.value = definition.default ?? {x: e.minimum, y: e.minimum};
    }, Slider, {
        x: definition.axes?.[0] ?? "X",
        y: definition.axes?.[1] ?? "Y",
    }),

    "temporal.vector.FloatVector|box": (parent, definition) => parent.createChild(VectorEditor, (e) => {
        e.minimum = definition.minimum ?? undefined;
        e.maximum = definition.maximum ?? undefined;
        e.step = definition.step ?? 0.1;
        e.value = definition.default ?? {x: e.minimum, y: e.minimum};
    }, NumberBox, {
        x: definition.axes?.[0] ?? "X",
        y: definition.axes?.[1] ?? "Y",
    }),

    "temporal.vector.FloatVector|slider": (parent, definition) => parent.createChild(VectorEditor, (e) => {
        e.minimum = definition.minimum ?? undefined;
        e.maximum = definition.maximum ?? undefined;
        e.step = definition.step ?? 0.1;
        e.value = definition.default ?? {x: e.minimum, y: e.minimum};
    }, Slider, {
        x: definition.axes?.[0] ?? "X",
        y: definition.axes?.[1] ?? "Y",
    }),

    "temporal.video_renderer.VideoRenderer": (parent, definition) => parent.createChild(VideoRendererEditor, (e) => {
        e.value = definition.default ?? {};
    }),
};
