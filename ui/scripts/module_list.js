import {Column} from "../scripts/base/column.js";
import {Dropdown} from "../scripts/base/dropdown.js";
import {ReorderableList} from "../scripts/base/reorderable_list.js";
import {Row} from "../scripts/base/row.js";
import {ToolButton} from "../scripts/base/tool_button.js";
import {Signal} from "../scripts/core/signal.js";
import {clearElement} from "../scripts/utils/dom.js";

export class ModuleList extends Column {
    constructor(moduleClass, names, definitions) {
        super();

        this.onValueChange = new Signal();

        this._moduleClass = moduleClass;
        this._names = names;
        this._definitions = definitions;
        this._value = [];

        this.createChild(Row, (e) => {
            let selectedModule = e.createChild(Dropdown, (e) => {
                e.choices = names;
                e.style.width = "100%";
            });

            e.createChild(ToolButton, (e) => {
                e.label = "+";
                e.onClick.connect(() => {
                    let module = this._createModule(selectedModule.value);
                    this._value.push(module.value);

                    this.onValueChange.fire(this._value);
                });
            });
        });

        this._list = this.createChild(ReorderableList, (e) => {
            e.onOrderChange.connect(() => {
                this._value = [...e.childNodes].map((node) => node.value);

                this.onValueChange.fire(this._value);
            });
        });
    }

    get value() {
        return this._value;
    }

    set value(value) {
        this._value = value;

        clearElement(this._list);

        for (let module of value) {
            this._createModule(module.__type__, module);
        }

        this.onValueChange.fire(value);
    }

    _createModule(type, initialValue = null) {
        return this._list.createChild(this._moduleClass, (e) => {
            e.label = this._names[type];

            if (initialValue) {
                e.value = initialValue;
            }

            e.onValueChange.connect(() => {
                this.onValueChange.fire(this._value);
            });
            e.onRemove.connect(() => {
                this._value = [...this._list.childNodes].map((node) => node.value);

                this.onValueChange.fire(this._value);
            });
        }, this._definitions[type]);
    }
}
customElements.define("module-list", ModuleList);
