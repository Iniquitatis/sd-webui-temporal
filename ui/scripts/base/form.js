import {Block} from "/scripts/base/block.js";
import {GroupBox} from "/scripts/base/group_box.js";
import {Widget} from "/scripts/core/widget.js";
import {createElement, defineElement} from "/scripts/utils/dom.js";

export class Form extends Widget {
    static tag = "ce-form";
    static css = `
        <self>.row {
            display: grid;
            gap: var(--layout-gap);
            grid-auto-columns: minmax(0, 1fr);
            grid-auto-flow: column;
        }

        <self>.column {
            display: flex;
            flex-direction: column;
            gap: var(--layout-gap);
        }

        <self> > .field {
            display: flex;
            flex-direction: column;
        }

        <self> > .field > ce-block {
            align-content: center;
            color: var(--hint-color);
            font-size: var(--hint-size);
            margin-bottom: var(--layout-small-gap);
        }
    `;

    constructor(isRow = false) {
        super();

        this.classList.add(isRow ? "row" : "column");
    }

    createField(name, tagOrClass, initializer, ...args) {
        let result = createElement(null, tagOrClass, initializer, ...args);

        if (result.canAttachTitle && result.canAttachTitle()) {
            result.attachTitle(name);
            result.formItem = result;
            this.appendChild(result);
        } else if (result.isComplexWidget && result.isComplexWidget()) {
            this.createChild(GroupBox, (e) => {
                e.label = name;
                result.formItem = e;
                // FIXME: Accesses private stuff
                e._fieldset.appendChild(result);
            });
        } else {
            this.createChild(Block, (e) => {
                e.classList.add("field");

                e.createChild(Block, (e) => {
                    e.innerText = name;
                });

                result.formItem = e;
                e.appendChild(result);
            });
        }

        return result;
    }

    createColumn(initializer) {
        return this.createChild(Form, initializer, false);
    }

    createRow(initializer) {
        return this.createChild(Form, initializer, true);
    }
}
defineElement(Form);
