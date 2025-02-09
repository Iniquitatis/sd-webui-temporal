import {Checkbox} from "../scripts/base/checkbox.js";
import {Column} from "../scripts/base/column.js";
import {NumberBox} from "../scripts/base/number_box.js";
import {Slider} from "../scripts/base/slider.js";
import {Row} from "../scripts/base/row.js";

export class VideoRendererEditor extends Column {
    constructor() {
        super();

        this.createChild(Slider, (e) => {
            e.label = "Frames per second";
            e.minimum = 1;
            e.maximum = 60;
            e.step = 1;
            e.value = 30;
        });

        this.createChild(Row, (e) => {
            e.createChild(NumberBox, (e) => {
                e.label = "First frame";
                e.minimum = 1;
                e.step = 1;
                e.value = 1;
            });

            e.createChild(NumberBox, (e) => {
                e.label = "Last frame";
                e.minimum = 0;
                e.step = 1;
                e.value = 1;
            });
        });

        this.createChild(NumberBox, (e) => {
            e.label = "Frame stride";
            e.minimum = 1;
            e.step = 1;
            e.value = 1;
        });

        this.createChild(Checkbox, (e) => {
            e.label = "Looping";
            e.value = false;
        });
    }
}
customElements.define("video-renderer-editor", VideoRendererEditor);
