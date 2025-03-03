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
import {GradientEditor} from "../scripts/gradient_editor.js";
import {ImageSourceEditor} from "../scripts/image_source_editor.js";
import {NoiseEditor} from "../scripts/noise_editor.js";
import {PatternEditor} from "../scripts/pattern_editor.js";
import {ProcessingParamsEditor} from "../scripts/processing_params_editor.js";
import {VideoRendererEditor} from "../scripts/video_renderer_editor.js";

export class ConfigurableParamEditor extends Widget {
    constructor(definition) {
        super();

        this._editor = null;

        switch (definition.type) {
            case "bool": {
                this._editor = this.createChild(Checkbox, (e) => {
                    e.value = definition.value;
                });
            } break;

            case "int": {
                this._editor = this.createChild(definition.ui_type == "slider" ? Slider : NumberBox, (e) => {
                    e.minimum = definition.minimum ?? undefined;
                    e.maximum = definition.maximum ?? undefined;
                    e.step = definition.step ?? 1;
                    e.value = definition.default ?? e.minimum;
                });
            } break;

            case "float": {
                this._editor = this.createChild(definition.ui_type == "slider" ? Slider : NumberBox, (e) => {
                    e.minimum = definition.minimum ?? undefined;
                    e.maximum = definition.maximum ?? undefined;
                    e.step = definition.step ?? 0.1;
                    e.value = definition.default ?? e.minimum;
                });
            } break;

            case "str": {
                if (definition.ui_type == "menu" || definition.ui_type == "radio") {
                    this._editor = this.createChild(definition.ui_type == "radio" ? Radio : Dropdown, (e) => {
                        e.choices = definition.choices ?? {"": ""};
                        e.value = definition.default ?? null;
                    });
                } else {
                    this._editor = this.createChild(definition.ui_type == "code" ? CodeArea : definition.ui_type == "area" ? TextArea : TextBox, (e) => {
                        e.value = definition.default ?? "";
                    });

                }
            } break;

            case "pathlib.Path":{
                this._editor = this.createChild(TextBox, (e) => {
                    e.value = definition.default ?? "";
                });
            } break;

            case "numpy.ndarray": {
                this._editor = this.createChild(ImageBox, (e) => {
                    e.channels = definition.channels ?? 3;
                });
            } break;

            case "temporal.color.Color": {
                this._editor = this.createChild(ColorPicker, (e) => {
                    e.value = definition.default ?? {r: 0.0, g: 0.0, b: 0.0, a: 1.0};
                }, definition.channels ?? 3);
            } break;

            case "temporal.gradient.Gradient": {
                this._editor = this.createChild(GradientEditor, (e) => {
                    e.value = definition.default ?? {};
                });
            } break;

            case "temporal.image_source.ImageSource": {
                this._editor = this.createChild(ImageSourceEditor, (e) => {
                    e.channels = definition.channels ?? 3;
                });
            } break;

            case "temporal.noise.Noise": {
                this._editor = this.createChild(NoiseEditor, (e) => {
                    e.value = definition.default ?? {};
                });
            } break;

            case "temporal.pattern.Pattern": {
                this._editor = this.createChild(PatternEditor, (e) => {
                    e.value = definition.default ?? {};
                });
            } break;

            case "temporal.processing_params.ProcessingParams": {
                this._editor = this.createChild(ProcessingParamsEditor, (e) => {
                    e.value = definition.default ?? {};
                });
            } break;

            case "temporal.vector.IntVector": {
                this._editor = this.createChild(VectorEditor, (e) => {
                    e.minimum = definition.minimum ?? undefined;
                    e.maximum = definition.maximum ?? undefined;
                    e.step = definition.step ?? 1;
                    e.value = definition.default ?? {x: e.minimum, y: e.minimum};
                }, definition.ui_type == "slider" ? Slider : NumberBox, {
                    x: definition.axes?.[0] ?? "X",
                    y: definition.axes?.[1] ?? "Y",
                });
            } break;

            case "temporal.vector.FloatVector": {
                this._editor = this.createChild(VectorEditor, (e) => {
                    e.minimum = definition.minimum ?? undefined;
                    e.maximum = definition.maximum ?? undefined;
                    e.step = definition.step ?? 0.1;
                    e.value = definition.default ?? {x: e.minimum, y: e.minimum};
                }, definition.ui_type == "slider" ? Slider : NumberBox, {
                    x: definition.axes?.[0] ?? "X",
                    y: definition.axes?.[1] ?? "Y",
                });
            } break;

            case "temporal.video_renderer.VideoRenderer": {
                this._editor = this.createChild(VideoRendererEditor, (e) => {
                    e.value = definition.default ?? {};
                });
            } break;

            default: {
                console.log(`WARNING: Unhandled type ${definition.type}`);
            } break;
        }

        if (this._editor) {
            this._editor.onValueChange.connect((value) => {
                this.onValueChange.fire(value);
            });
        }

        this.onValueChange = new Signal();
    }

    get value() {
        return this._editor.value;
    }

    set value(value) {
        this._editor.value = value;
    }
}
customElements.define("configurable-param-editor", ConfigurableParamEditor);
