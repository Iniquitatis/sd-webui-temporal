import {Widget} from "/scripts/core/widget.js";
import {defineElement} from "/scripts/utils/dom.js";

export class Row extends Widget {
    static tag = "ce-row";
    static css = `
        <self> {
            display: flex;
            flex-direction: row;
            gap: var(--layout-gap);
        }

        <self> > * {
            width: 100%;
        }
    `;
}
defineElement(Row);
