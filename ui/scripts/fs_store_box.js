import {Dropdown} from "../scripts/base/dropdown.js";
import {Row} from "../scripts/base/row.js";
import {ToolButton} from "../scripts/base/tool_button.js";
import {Signal} from "../scripts/core/signal.js";
import {Widget} from "../scripts/core/widget.js";
import {getRequest, postRequest} from "../scripts/utils/requests.js";

export class FSStoreBox extends Widget {
    constructor(store, features) {
        super();

        this.onValueChange = new Signal();
        this.onRefresh = new Signal();
        this.onLoad = new Signal();
        this.onSave = new Signal();
        this.onRename = new Signal();
        this.onDelete = new Signal();

        this._dropdown = this.createChild(Dropdown, (e) => {
            e.onValueChange.connect((value) => {
                this.onValueChange.fire(value);
            });

            e.createChild(Row, (e) => {
                e.createChild(ToolButton, (e) => {
                    e.label = "\u{1f504}\u{fe0e}";
                    e.style.display = features.includes("refresh") ? null : "none";
                    e.onClick.connect(async () => {
                        await postRequest("/temporal/fs_operation", {
                            "store": store,
                            "operation": "refresh",
                        });

                        await getRequest(`/temporal/${store}`, (result) => {
                            this.entries = result;
                        });

                        this.onRefresh.fire();
                    });
                });

                e.createChild(ToolButton, (e) => {
                    e.label = "\u{1f4c2}\u{fe0e}";
                    e.style.display = features.includes("load") ? null : "none";
                    e.onClick.connect(() => {
                        // TODO
                        this.onLoad.fire(this._dropdown.value);
                    });
                });

                e.createChild(ToolButton, (e) => {
                    e.label = "\u{1f4be}\u{fe0e}";
                    e.style.display = features.includes("save") ? null : "none";
                    e.onClick.connect(() => {
                        // TODO
                        this.onSave.fire(this._dropdown.value);
                    });
                });

                e.createChild(ToolButton, (e) => {
                    e.label = "\u{270e}\u{fe0f}";
                    e.style.display = features.includes("rename") ? null : "none";
                    e.onClick.connect(async () => {
                        let oldName = this._dropdown.value;
                        let newName = window.prompt("Enter new name:", oldName);
                        if (!newName) return;

                        await postRequest("/temporal/fs_operation", {
                            "store": store,
                            "operation": "rename",
                            "args": {
                                "old_name": oldName,
                                "new_name": newName,
                            },
                        });

                        await getRequest(`/temporal/${store}`, (result) => {
                            this.entries = result;
                        });

                        this.onRename.fire(oldName, newName);
                    });
                });

                e.createChild(ToolButton, (e) => {
                    e.label = "\u{274c}\u{fe0e}";
                    e.style.display = features.includes("delete") ? null : "none";
                    e.onClick.connect(async () => {
                        let name = this._dropdown.value;

                        if (!confirm(`Are you sure you want to delete "${name}"?`)) return;

                        await postRequest("/temporal/fs_operation", {
                            "store": store,
                            "operation": "delete",
                            "args": {
                                "name": name,
                            },
                        });

                        await getRequest(`/temporal/${store}`, (result) => {
                            this.entries = result;
                        });

                        this.onDelete.fire(name);
                    });
                });
            });
        });
    }

    get entries() {
        return this._dropdown.choices;
    }

    get label() {
        return this._dropdown.label;
    }

    get value() {
        return this._dropdown.value;
    }

    set entries(value) {
        let choices = {};

        for (let entry of value) {
            choices[entry] = entry;
        }

        this._dropdown.choices = choices;
    }

    set label(value) {
        this._dropdown.label = value;
    }

    set value(value) {
        this._dropdown.value = value;

        this.onValueChange.fire(value);
    }
}
customElements.define("fs-store-box", FSStoreBox);
