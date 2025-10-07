import {MediaBox} from "/scripts/base/media_box.js";
import {VideoWidget} from "/scripts/base/video_widget.js";

export class VideoBox extends MediaBox {
    constructor(features = []) {
        super(VideoWidget, "video/*", features);
    }
}
customElements.define("ce-video-box", VideoBox);
