import {Widget} from "/scripts/core/widget.js";
import {defineElement} from "/scripts/utils/dom.js";

export class Column extends Widget {
    static tag = "ce-column";
    static css = `
        <self> {
            display: flex;
            flex-direction: column;
            gap: var(--layout-gap);
            width: 100%;
        }
    `;
}
defineElement(Column);
