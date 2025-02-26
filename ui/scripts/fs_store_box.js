import {Dropdown} from "../scripts/base/dropdown.js";
import {Row} from "../scripts/base/row.js";
import {ToolButton} from "../scripts/base/tool_button.js";
import {Signal} from "../scripts/core/signal.js";
import {getRequest, postRequest} from "../scripts/utils/requests.js";

export class FSStoreBox extends Row {
    constructor(store, features) {
        super();

        this.saveCallback = null;

        this.onValueChange = new Signal();
        this.onRefresh = new Signal();
        this.onLoad = new Signal();
        this.onSave = new Signal();
        this.onRename = new Signal();
        this.onDelete = new Signal();

        this._dropdown = this.createChild(Dropdown, (e) => {
            e.style.width = "100%";

            e.onValueChange.connect((value) => {
                this.onValueChange.fire(value);
            });
        });

        this.createChild(Row, (e) => {
            e.style.gap = "calc(var(--layout-gap) / 2)";

            e.createChild(ToolButton, (e) => {
                e.label = "\u{f021}";
                e.style.display = features.includes("refresh") ? null : "none";
                e.onClick.connect(async () => {
                    await postRequest("/temporal/fs_operation", {
                        "store": store,
                        "operation": "refresh",
                    });

                    this.entries = await getRequest(`/temporal/${store}`);

                    this.onRefresh.fire();
                });
            });

            e.createChild(ToolButton, (e) => {
                e.label = "\u{f07c}";
                e.style.display = features.includes("load") ? null : "none";
                e.onClick.connect(async () => {
                    this.onLoad.fire(await postRequest("/temporal/fs_operation", {
                        "store": store,
                        "operation": "load",
                        "args": {
                            "name": this._dropdown.value,
                        },
                    }));
                });
            });

            e.createChild(ToolButton, (e) => {
                e.label = "\u{f0c7}";
                e.style.display = features.includes("save") ? null : "none";
                e.onClick.connect(async () => {
                    if (!this.saveCallback) return;

                    await postRequest("/temporal/fs_operation", {
                        "store": store,
                        "operation": "save",
                        "args": {
                            // FIXME
                            "name": "DEFAULT",
                            "data": this.saveCallback(),
                        },
                    });

                    this.entries = await getRequest(`/temporal/${store}`);

                    this.onSave.fire();
                });
            });

            e.createChild(ToolButton, (e) => {
                e.label = "\u{f31c}";
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

                    this.entries = await getRequest(`/temporal/${store}`);

                    this.onRename.fire(oldName, newName);
                });
            });

            e.createChild(ToolButton, (e) => {
                e.label = "\u{f2ed}";
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

                    this.entries = await getRequest(`/temporal/${store}`);

                    this.onDelete.fire(name);
                });
            });
        });
    }

    get entries() {
        return this._dropdown.choices;
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

    set value(value) {
        this._dropdown.value = value;

        this.onValueChange.fire(value);
    }
}
customElements.define("fs-store-box", FSStoreBox);
