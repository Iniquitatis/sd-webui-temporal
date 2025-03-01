import {Form} from "../scripts/base/form.js";
import {FieldManager} from "../scripts/core/field_manager.js";
import {Signal} from "../scripts/core/signal.js";
import {mapObject} from "../scripts/utils/object.js";
import {AnimationEditor} from "../scripts/animation_editor.js";
import {ModuleList} from "../scripts/module_list.js";
import {PipelineModuleEditor} from "../scripts/pipeline_module_editor.js";
import {pipelineModules} from "../scripts/shared_data.js";

export class PipelineEditor extends Form {
    constructor() {
        super();

        this.onValueChange = new Signal();

        this._manager = new FieldManager(this.onValueChange);

        this.createField("Add module", ModuleList, (e) => {
            this._manager.manage(e, "modules");
        }, PipelineModuleEditor, mapObject(pipelineModules, (_, module) => {
            for (let [start, icon] of Object.entries({
                "temporal.pipeline_modules.filtering": "\u{f890}",
                "temporal.pipeline_modules.measuring": "\u{f201}",
                "temporal.pipeline_modules.neural": "\u{e0c6}",
                "temporal.pipeline_modules.painting": "\u{f1fc}",
                "temporal.pipeline_modules.temporal": "\u{f017}",
                "temporal.pipeline_modules.tool": "\u{f0ad}",
            })) {
                if (module.type.startsWith(start)) {
                    return `${icon} ${module.name}`;
                }
            }

            return `"\u{f013}" ${module.name}`;
        }), pipelineModules);

        this.createField("Animation", AnimationEditor, (e) => {
            this._manager.manage(e, "animation");
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
