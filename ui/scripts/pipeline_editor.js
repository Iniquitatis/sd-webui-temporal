import {ToolButton} from "../scripts/core/buttons.js";
import {Accordion, Column, ReorderableList, Row} from "../scripts/core/layout.js";
import {Signal} from "../scripts/core/signal.js";
import {EnumEditor, NumberEditor} from "../scripts/core/value_editors.js";
import {getObjectKeyByIndex, mapObject} from "../scripts/utils/object.js";
import {InitialNoiseEditor} from "../scripts/initial_noise_editor.js";
import {PipelineModuleEditor} from "../scripts/pipeline_module_editor.js";
import {modules} from "../scripts/test_data.js";

export class PipelineEditor extends Column {
    constructor() {
        super();

        let pipeline = {modules: []};

        this.createChild(Accordion, (e) => {
            e.label = "Initial noise";

            e.createChild(InitialNoiseEditor, (e) => {
                e.onValueChange.connect((value) => {
                    pipeline.initial_noise = value;

                    this.onValueChange.fire(pipeline);
                });
            });
        });

        this.createChild(NumberEditor, (e) => {
            e.label = "Parallel";
            e.variant = "box";
            e.minimum = 1;
            e.maximum = 16;
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
            e.choices = mapObject(modules, (_, module) => `${module.icon} ${module.name}`);
            e.value = getObjectKeyByIndex(e.choices, 0);

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
                    }, key, modules[key]);

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
        this.onValueChange.connect(value => console.log(value));
    }

    get modules() {
        return [...this._list.childNodes].map((node) => node.module);
    }
}
customElements.define("pipeline-editor", PipelineEditor);
