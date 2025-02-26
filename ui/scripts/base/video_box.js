import {MediaBox} from "../../scripts/base/media_box.js";
import {VideoWidget} from "../../scripts/base/video_widget.js";

export class VideoBox extends MediaBox {
    constructor() {
        super(VideoWidget, "video/*");
    }
}
customElements.define("video-box", VideoBox);
