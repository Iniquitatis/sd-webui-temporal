import {Column} from "../scripts/base/column.js";
import {Dropdown} from "../scripts/base/dropdown.js";
import {NumberBox} from "../scripts/base/number_box.js";
import {ReorderableList} from "../scripts/base/reorderable_list.js";
import {ToolButton} from "../scripts/base/tool_button.js";
import {FieldManager} from "../scripts/core/field_manager.js";
import {Signal} from "../scripts/core/signal.js";
import {clearElement} from "../scripts/utils/dom.js";
import {mapObject} from "../scripts/utils/object.js";
import {PipelineModuleEditor} from "../scripts/pipeline_module_editor.js";
import {pipelineModules} from "../scripts/test_data.js";

class PipelineModuleList extends ReorderableList {
    constructor() {
        super();

        this.onValueChange = new Signal();

        this.onOrderChange.connect(() => {
            this._value = [...this.childNodes].map((node) => node.value);

            this.onValueChange.fire(this._value);
        });
    }

    get value() {
        return this._value;
    }

    set value(value) {
        this._value = value;

        clearElement(this);

        for (let module of value) {
            this._createModule(module.id, module);
        }

        this.onValueChange.fire(value);
    }

    addModule(id, initialValue = null) {
        let result = this._createModule(id, initialValue);
        this._value.push(result.value);
        return result;
    }

    _createModule(id, initialValue = null) {
        return this.createChild(PipelineModuleEditor, (e) => {
            if (initialValue) {
                e.value = initialValue;
            }

            e.onValueChange.connect(() => {
                this.onValueChange.fire(this._value);
            });
            e.onRemove.connect(() => {
                this._value = [...this.childNodes].map((node) => node.value);

                this.onValueChange.fire(this._value);
            });
        }, id, pipelineModules[id]);
    }
}
customElements.define("pipeline-module-list", PipelineModuleList);

export class PipelineEditor extends Column {
    constructor() {
        super();

        this.onValueChange = new Signal();

        this._manager = new FieldManager(this.onValueChange);

        this.createChild(NumberBox, (e) => {
            e.label = "Parallel";
            e.minimum = 1;
            e.step = 1;
            e.value = 1;
            this._manager.manage(e, "parallel");
        });

        let selectedModule = this.createChild(Dropdown, (e) => {
            e.label = "Add module";
            e.choices = mapObject(pipelineModules, (_, module) => `${module.icon} ${module.name}`);

            e.createChild(ToolButton, (e) => {
                e.label = "+";
                e.onClick.connect(() => {
                    this._list.addModule(selectedModule.value);

                    this.onValueChange.fire(this._value);
                });
            });
        });

        this._list = this.createChild(PipelineModuleList, (e) => {
            this._manager.manage(e, "modules");
        });
    }

    get value() {
        return this._manager.value;
    }

    set value(value) {
        this._manager.value = value;
    }
}
customElements.define("pipeline-editor", PipelineEditor);
