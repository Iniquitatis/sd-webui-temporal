import {BoolEditor} from "../scripts/base/bool_editor.js";
import {Column} from "../scripts/base/column.js";
import {NumberEditor} from "../scripts/base/number_editor.js";
import {Row} from "../scripts/base/row.js";

export class VideoRendererEditor extends Column {
    constructor() {
        super();

        this.createChild(NumberEditor, (e) => {
            e.label = "Frames per second";
            e.variant = "slider";
            e.minimum = 1;
            e.maximum = 60;
            e.step = 1;
            e.value = 30;
        });

        this.createChild(Row, (e) => {
            e.createChild(NumberEditor, (e) => {
                e.label = "First frame";
                e.variant = "box";
                e.minimum = 1;
                e.maximum = 2 ** 32 - 1;
                e.step = 1;
                e.value = 1;
            });

            e.createChild(NumberEditor, (e) => {
                e.label = "Last frame";
                e.variant = "box";
                e.minimum = 0;
                e.maximum = 2 ** 32 - 1;
                e.step = 1;
                e.value = 1;
            });
        });

        this.createChild(NumberEditor, (e) => {
            e.label = "Frame stride";
            e.variant = "box";
            e.minimum = 1;
            e.maximum = 2 ** 32 - 1;
            e.step = 1;
            e.value = 1;
        });

        this.createChild(BoolEditor, (e) => {
            e.label = "Looping";
            e.value = false;
        });
    }
}
customElements.define("video-renderer-editor", VideoRendererEditor);
