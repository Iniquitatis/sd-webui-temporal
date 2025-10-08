import {Block} from "/scripts/base/block.js";
import {MultiStateToggle} from "/scripts/base/multi_state_toggle.js";
import {Widget} from "/scripts/core/widget.js";
import {createElement, defineElement} from "/scripts/utils/dom.js";

export class Accordion extends Widget {
    static tag = "ce-accordion";
    static css = `
        <self> {
            border: var(--thin-border);
            border-radius: var(--corners);
        }

        <self> > ce-block:nth-of-type(1) {
            align-items: center;
            display: flex;
            flex-direction: row;
            justify-content: space-between;
        }

        <self> > ce-block:nth-of-type(1) > ce-block {
            align-content: center;
            height: var(--widget-height);
            padding: 0 var(--horizontal-padding);
            user-select: none;
            width: 100%;
        }

        <self> > ce-block:nth-of-type(2) {
            padding: var(--layout-padding);
            padding-top: 0;
        }
    `;

    constructor() {
        super();

        this._header = super.createChild(Block, (e) => {
            this._label = e.createChild(Block, (e) => {
                e.addEventListener("click", () => {
                    this._openToggle.nextState();
                });
            });

            this._openToggle = e.createChild(MultiStateToggle, (e) => {
                e.states = {opened: "\u{f0d7}", closed: "\u{f0d9}"};
                e.value = "closed";
                e.onValueChange.connect((value) => {
                    this._content.visible = value == "opened";
                });
            });
        });

        this._content = super.createChild(Block, (e) => {
            e.visible = false;
        });
    }

    get label() {
        return this._label.innerText;
    }

    set label(value) {
        this._label.innerText = value;
    }

    createBeforeLabel(tagOrClass, initializer, ...args) {
        this._header.insertBefore(createElement(null, tagOrClass, initializer, ...args), this._label);
    }

    createAfterLabel(tagOrClass, initializer, ...args) {
        this._header.insertBefore(createElement(null, tagOrClass, initializer, ...args), this._openToggle);
    }

    createChild(tagOrClass, initializer, ...args) {
        return this._content.createChild(tagOrClass, initializer, ...args);
    }
}
defineElement(Accordion);
