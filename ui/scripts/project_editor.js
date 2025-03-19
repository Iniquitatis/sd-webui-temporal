import {Button} from "../scripts/base/button.js";
import {Column} from "../scripts/base/column.js";
import {Tabs} from "../scripts/base/tabs.js";
import {FieldManager} from "../scripts/core/field_manager.js";
import {Signal} from "../scripts/core/signal.js";
import {GeneralDataEditor} from "../scripts/general_data_editor.js";
import {PipelineEditor} from "../scripts/pipeline_editor.js";

export class ProjectEditor extends Tabs {
    constructor() {
        super();

        this.onValueChange = new Signal();

        this._manager = new FieldManager(this.onValueChange);

        this.createTab("General", GeneralDataEditor, (e) => {
            this._manager.manage(e, "general");
        });

        this.createTab("Pipeline", PipelineEditor, (e) => {
            this._manager.manage(e, "pipeline");
        });

        this.createTab("Tools", Column, (e) => {
            e.createChild(Button, (e) => {
                e.label = "Delete session data";
            });
        });
    }

    get value() {
        return this._manager.value;
    }

    set value(value) {
        this._manager.value = value;
    }
}
customElements.define("project-editor", ProjectEditor);
