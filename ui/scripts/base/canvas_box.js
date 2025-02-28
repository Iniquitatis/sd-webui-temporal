import {CanvasWidget} from "../../scripts/base/canvas_widget.js";
import {MediaBox} from "../../scripts/base/media_box.js";

export class CanvasBox extends MediaBox {
    constructor() {
        super(CanvasWidget, "image/*");
    }
}
customElements.define("canvas-box", CanvasBox);
