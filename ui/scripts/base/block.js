import {Widget} from "/scripts/core/widget.js";
import {defineElement} from "/scripts/utils/dom.js";

export class Block extends Widget {
    static tag = "ce-block";
    static css = `
        <self> {
            display: block;
        }
    `;
}
defineElement(Block);
