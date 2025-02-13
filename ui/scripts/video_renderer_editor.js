import {Checkbox} from "../scripts/base/checkbox.js";
import {Column} from "../scripts/base/column.js";
import {NumberBox} from "../scripts/base/number_box.js";
import {Slider} from "../scripts/base/slider.js";
import {Row} from "../scripts/base/row.js";
import {FieldManager} from "../scripts/core/field_manager.js";
import {Signal} from "../scripts/core/signal.js";
import {mapObject} from "../scripts/utils/object.js";
import {ModuleList} from "../scripts/module_list.js";
import {videoFilters} from "../scripts/shared_data.js";
import {VideoFilterEditor} from "../scripts/video_filter_editor.js";

export class VideoRendererEditor extends Column {
    constructor() {
        super();

        this.onValueChange = new Signal();

        this._manager = new FieldManager(this.onValueChange);

        this.createChild(Slider, (e) => {
            e.label = "Frames per second";
            e.minimum = 1;
            e.maximum = 60;
            e.step = 1;
            e.value = 30;
            this._manager.manage(e, "fps");
        });

        this.createChild(Row, (e) => {
            e.createChild(NumberBox, (e) => {
                e.label = "First frame";
                e.minimum = 1;
                e.step = 1;
                e.value = 1;
                this._manager.manage(e, "first_frame");
            });

            e.createChild(NumberBox, (e) => {
                e.label = "Last frame";
                e.minimum = 0;
                e.step = 1;
                e.value = 0;
                this._manager.manage(e, "last_frame");
            });
        });

        this.createChild(NumberBox, (e) => {
            e.label = "Frame stride";
            e.minimum = 1;
            e.step = 1;
            e.value = 1;
            this._manager.manage(e, "frame_stride");
        });

        this.createChild(Checkbox, (e) => {
            e.label = "Looping";
            e.value = false;
            this._manager.manage(e, "looping");
        });

        this.createChild(ModuleList, (e) => {
            this._manager.manage(e, "filters");
        }, "Add filter", VideoFilterEditor, mapObject(videoFilters, (_, filter) => filter.name), videoFilters);
    }

    get value() {
        return this._manager.value;
    }

    set value(value) {
        this._manager.value = value;
    }
}
customElements.define("video-renderer-editor", VideoRendererEditor);
