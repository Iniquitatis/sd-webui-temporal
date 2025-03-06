import {ImageWidget} from "../../scripts/base/image_widget.js";
import {MediaBox} from "../../scripts/base/media_box.js";

export class ImageBox extends MediaBox {
    constructor(features = []) {
        super(ImageWidget, "image/*", features);
    }
}
customElements.define("image-box", ImageBox);
