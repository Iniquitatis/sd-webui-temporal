import {Button} from "../scripts/base/button.js";
import {Checkbox} from "../scripts/base/checkbox.js";
import {Form} from "../scripts/base/form.js";
import {NumberBox} from "../scripts/base/number_box.js";
import {Row} from "../scripts/base/row.js";
import {Slider} from "../scripts/base/slider.js";
import {VideoBox} from "../scripts/base/video_box.js";
import {FieldManager} from "../scripts/core/field_manager.js";
import {Signal} from "../scripts/core/signal.js";
import {mapObject} from "../scripts/utils/object.js";
import {postRequest} from "../scripts/utils/requests.js";
import {ModuleList} from "../scripts/module_list.js";
import {videoFilters} from "../scripts/shared_data.js";
import {VideoFilterEditor} from "../scripts/video_filter_editor.js";

export class VideoRendererEditor extends Form {
    constructor() {
        super();

        this.onValueChange = new Signal();

        this._manager = new FieldManager(this.onValueChange);

        this.createField("Frames per second", Slider, (e) => {
            e.minimum = 1;
            e.maximum = 60;
            e.step = 1;
            e.value = 30;
            this._manager.manage(e, "fps");
        });

        this.createRow((e) => {
            e.createField("First frame", NumberBox, (e) => {
                e.minimum = 1;
                e.step = 1;
                e.value = 1;
                this._manager.manage(e, "first_frame");
            });

            e.createField("Last frame", NumberBox, (e) => {
                e.minimum = 0;
                e.step = 1;
                e.value = 0;
                this._manager.manage(e, "last_frame");
            });
        });

        this.createField("Frame stride", NumberBox, (e) => {
            e.minimum = 1;
            e.step = 1;
            e.value = 1;
            this._manager.manage(e, "frame_stride");
        });

        this.createField("Looping", Checkbox, (e) => {
            e.value = false;
            this._manager.manage(e, "looping");
        });

        this.createField("Add filter", ModuleList, (e) => {
            this._manager.manage(e, "filters");
        }, VideoFilterEditor, mapObject(videoFilters, (_, filter) => filter.name), videoFilters);

        this._buttons = this.createChild(Row, (e) => {
            for (let type of ["draft", "final"]) {
                e.createChild(Button, (e) => {
                    e.label = `Render ${type}`;
                    e.style.width = "100%";
                    e.onClick.connect(async () => {
                        this._buttons.classList.add("disabled");

                        this._video.value = await postRequest("/temporal/render_video", {
                            "type": type,
                            "data": this._manager.value,
                        });

                        this._buttons.classList.remove("disabled");
                    });
                });
            }
        });

        this._video = this.createChild(VideoBox);
    }

    get value() {
        return this._manager.value;
    }

    set value(value) {
        this._manager.value = value;
    }
}
customElements.define("video-renderer-editor", VideoRendererEditor);
