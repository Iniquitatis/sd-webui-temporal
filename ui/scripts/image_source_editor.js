import {Form} from "../scripts/base/form.js";
import {ImageBox} from "../scripts/base/image_box.js";
import {Radio} from "../scripts/base/radio.js";
import {VideoBox} from "../scripts/base/video_box.js";
import {FieldManager} from "../scripts/core/field_manager.js";
import {Signal} from "../scripts/core/signal.js";

export class ImageSourceEditor extends Form {
    constructor() {
        super();

        this.onValueChange = new Signal();

        this._manager = new FieldManager(this.onValueChange);

        this.createField("Type", Radio, (e) => {
            e.choices = {
                "image": "Image",
                "initial_image": "Initial image",
                "video": "Video",
            };
            e.value = "image";
            e.onValueChange.connect((value) => {
                this._imageBox.visible = value == "image";
                this._videoBox.visible = value == "video";

                if (!this._imageBox.visible) {
                    this.onValueChange.withDisabled(() => {
                        this._imageBox.value = null;
                    });
                }

                if (!this._videoBox.visible) {
                    this.onValueChange.withDisabled(() => {
                        this._videoBox.value = null;
                    });
                }
            });
            this._manager.manage(e, "type");
        });

        this._imageBox = this.createChild(ImageBox, (e) => {
            e.visible = true;
            this._manager.manage(e, "value");
        }, ["clear", "download", "fullscreen", "upload"]);

        this._videoBox = this.createChild(VideoBox, (e) => {
            e.visible = false;
            this._manager.manage(e, "value");
        }, ["clear", "download", "fullscreen", "upload"]);
    }

    get value() {
        return this._manager.value;
    }

    set value(value) {
        this._manager.value = value;
    }

    isComplexWidget() {
        return true;
    }
}
customElements.define("image-source-editor", ImageSourceEditor);
