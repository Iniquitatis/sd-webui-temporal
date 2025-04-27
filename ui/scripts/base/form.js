import {Block} from "../../scripts/base/block.js";
import {GroupBox} from "../../scripts/base/group_box.js";
import {Widget} from "../../scripts/core/widget.js";
import {createElement} from "../../scripts/utils/dom.js";

export class Form extends Widget {
    constructor(isRow = false) {
        super();

        if (isRow) {
            this.style.display = "grid";
            this.style.gap = "var(--layout-gap)";
            this.style.gridAutoColumns = "minmax(0, 1fr)";
            this.style.gridAutoFlow = "column";
        } else {
            this.style.display = "flex";
            this.style.flexDirection = "column";
            this.style.gap = "var(--layout-gap)";
        }
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
                // FIXME: Accesses private stuff
                result.formItem = e;
                e._fieldset.appendChild(result);
            });
        } else {
            this.createChild(Block, (e) => {
                e.style.display = "flex";
                e.style.flexDirection = "column";

                e.createChild(Block, (e) => {
                    e.innerText = name;
                    e.style.alignContent = "center";
                    e.style.color = "var(--hint-color)";
                    e.style.fontSize = "0.9rem";
                    e.style.marginBottom = "var(--layout-small-gap)";
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
customElements.define("custom-form", Form);
