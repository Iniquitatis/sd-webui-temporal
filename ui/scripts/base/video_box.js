import {MediaBox} from "/scripts/base/media_box.js";
import {VideoWidget} from "/scripts/base/video_widget.js";
import {defineElement} from "/scripts/utils/dom.js";

export class VideoBox extends MediaBox {
    static tag = "ce-video-box";

    constructor(features = []) {
        super(VideoWidget, "video/*", features);
    }
}
defineElement(VideoBox);
