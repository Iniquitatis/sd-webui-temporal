import {Column} from "../scripts/base/column.js";
import {ImageBox} from "../scripts/base/image_box.js";
import {Radio} from "../scripts/base/radio.js";
import {VideoBox} from "../scripts/base/video_box.js";
import {FieldManager} from "../scripts/core/field_manager.js";
import {Signal} from "../scripts/core/signal.js";

export class ImageSourceEditor extends Column {
    constructor() {
        super();

        this.onValueChange = new Signal();

        this._manager = new FieldManager(this.onValueChange);

        this.createChild(Radio, (e) => {
            e.label = "Type";
            e.choices = {
                "image": "Image",
                "initial_image": "Initial image",
                "video": "Video",
            };
            e.value = "image";
            e.onValueChange.connect((value) => {
                this._imageBox.style.display = value == "image" ? null : "none";
                this._videoBox.style.display = value == "video" ? null : "none";

                if (this._imageBox.style.display == "none") {
                    this.onValueChange.withDisabled(() => {
                        this._imageBox.value = null;
                    });
                }

                if (this._videoBox.style.display == "none") {
                    this.onValueChange.withDisabled(() => {
                        this._videoBox.value = null;
                    });
                }
            });
            this._manager.manage(e, "type");
        });

        this._imageBox = this.createChild(ImageBox, (e) => {
            e.label = "Image";
            this._manager.manage(e, "value");
        });

        this._videoBox = this.createChild(VideoBox, (e) => {
            e.label = "Video";
            e.style.display = "none";
            this._manager.manage(e, "value");
        });
    }

    get value() {
        return this._manager.value;
    }

    set value(value) {
        this._manager.value = value;
    }
}
customElements.define("image-source-editor", ImageSourceEditor);
