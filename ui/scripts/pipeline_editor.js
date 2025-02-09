import {Column} from "../scripts/base/column.js";
import {EnumEditor} from "../scripts/base/enum_editor.js";
import {NumberEditor} from "../scripts/base/number_editor.js";
import {ReorderableList} from "../scripts/base/reorderable_list.js";
import {ToolButton} from "../scripts/base/tool_button.js";
import {Signal} from "../scripts/core/signal.js";
import {mapObject} from "../scripts/utils/object.js";
import {PipelineModuleEditor} from "../scripts/pipeline_module_editor.js";
import {pipelineModules} from "../scripts/test_data.js";

export class PipelineEditor extends Column {
    constructor() {
        super();

        let pipeline = {modules: []};

        this.createChild(NumberEditor, (e) => {
            e.label = "Parallel";
            e.variant = "box";
            e.minimum = 1;
            e.step = 1;
            e.value = 1;
            e.onValueChange.connect((value) => {
                pipeline.parallel = value;

                this.onValueChange.fire(pipeline);
            });
        });

        let selectedModule = this.createChild(EnumEditor, (e) => {
            e.label = "Add module";
            e.variant = "menu";
            e.choices = mapObject(pipelineModules, (_, module) => `${module.icon} ${module.name}`);

            e.createChild(ToolButton, (e) => {
                e.label = "+";
                e.onClick.connect(() => {
                    let key = selectedModule.value;

                    this._list.createChild(PipelineModuleEditor, (e) => {
                        pipeline.modules.push(e.module);

                        e.onValueChange.connect(() => {
                            this.onValueChange.fire(pipeline);
                        });
                        e.onRemove.connect(() => {
                            pipeline.modules = this.modules;

                            this.onValueChange.fire(pipeline);
                        });
                    }, key, pipelineModules[key]);

                    this.onValueChange.fire(pipeline);
                });
            });
        });

        this._list = this.createChild(ReorderableList);
        this._list.onOrderChange.connect(() => {
            pipeline.modules = this.modules;

            this.onValueChange.fire(pipeline);
        });

        this.onValueChange = new Signal();

        // FIXME: Temporary
        this.onValueChange.connect((value) => console.log(value));
    }

    get modules() {
        return [...this._list.childNodes].map((node) => node.module);
    }
}
customElements.define("pipeline-editor", PipelineEditor);
