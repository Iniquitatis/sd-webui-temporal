import {Button} from "../../scripts/base/button.js";
import {Column} from "../../scripts/base/column.js";
import {Dropdown} from "../../scripts/base/dropdown.js";
import {ReorderableList} from "../../scripts/base/reorderable_list.js";
import {Row} from "../../scripts/base/row.js";
import {ToolButton} from "../../scripts/base/tool_button.js";
import {Signal} from "../../scripts/core/signal.js";
import {clearElement, createElement} from "../../scripts/utils/dom.js";
import {deepCopy} from "../../scripts/utils/object.js";

class ListEditorBase extends Column {
    constructor(elementClass) {
        super();

        this.onValueChange = new Signal();

        this._elementClass = elementClass;
        this._value = [];

        this._list = this.createChild(ReorderableList, (e) => {
            e.visible = false;
            e.onOrderChange.connect(() => {
                this._rereadValue();
                this.onValueChange.fire(this._value);
            });
        });
    }

    get value() {
        return this._value;
    }

    set value(value) {
        this._value = value;
        this._rebuildList();
        this.onValueChange.fire(value);
    }

    getArgsFromItem(item) {
        return [];
    }

    _createEditor(...args) {
        return this._list.createChild(this._elementClass, (e) => {
            e.onValueChange.connect(() => {
                this._rereadValue();
                this.onValueChange.fire(this._value);
            });

            if (e.onDuplicateRequest) {
                e.onDuplicateRequest.connect((value = null) => {
                    let editor = this._createEditor(...args);
                    editor.onValueChange.withDisabled(() => editor.value = value ?? deepCopy(e.value));
                    this.onValueChange.fire(this._value);
                });
            }

            if (e.onRemoveRequest) {
                e.onRemoveRequest.connect(() => {
                    this._list.removeChild(e);
                    this._rereadValue();
                    this.onValueChange.fire(this._value);
                });
            }
        }, ...args);
    }

    _createItem(...args) {
        let element = this._createEditor(...args);
        this._value.push(element.value);
        this.onValueChange.fire(this._value);
        return element;
    }

    _rebuildList() {
        clearElement(this._list);

        for (let item of this._value) {
            let editor = this._createEditor(...this.getArgsFromItem(item));
            editor.onValueChange.withDisabled(() => editor.value = item);
        }
    }

    _rereadValue() {
        this._value.splice(0);
        this._value.push(...[...this._list.childNodes].map((node) => node.value));
    }
}

export class ListEditor extends ListEditorBase {
    constructor(elementClass) {
        super(elementClass);

        this._addButton = this.insertBefore(createElement(null, Button, (e) => {
            e.label = "\u{e59e} Add item";
            e.onClick.connect(() => {
                this._createItem(...this.getDefaultArgs());
            });
        }), this._list);
    }

    get addLabel() {
        return this._addButton.label;
    }

    set addLabel(value) {
        this._addButton.label = value;
    }

    getDefaultArgs() {
        return [];
    }
}
customElements.define("list-editor", ListEditor);

export class ChoiceListEditor extends ListEditorBase {
    constructor(elementClass, choices) {
        super(elementClass);

        this.insertBefore(createElement(null, Row, (e) => {
            e.style.gap = "var(--layout-small-gap)";

            let choice = e.createChild(Dropdown, (e) => {
                e.choices = choices;
                e.style.width = "100%";
            });

            e.createChild(ToolButton, (e) => {
                e.label = "\u{e59e}";
                e.onClick.connect(() => {
                    this._createItem(...this.getArgsFromChoice(choice.value));
                });
            });
        }), this._list);
    }

    getArgsFromChoice(choice) {
        return [];
    }
}
customElements.define("typed-list-editor", ChoiceListEditor);
