import {Button} from "/scripts/base/button.js";
import {defineElement} from "/scripts/utils/dom.js";

export class ToolButton extends Button {
    static tag = "ce-tool-button";
    static css = `
        <self> {
            max-width: var(--widget-height);
            min-width: var(--widget-height);
        }
    `;
}
defineElement(ToolButton);
