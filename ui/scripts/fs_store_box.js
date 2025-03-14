import {Dropdown} from "../scripts/base/dropdown.js";
import {Row} from "../scripts/base/row.js";
import {ToolButton} from "../scripts/base/tool_button.js";
import {Signal} from "../scripts/core/signal.js";
import {postRequest} from "../scripts/utils/requests.js";

export class FSStoreBox extends Row {
    constructor(store, features) {
        super();

        this.saveCallback = null;

        this.onValueChange = new Signal();
        this.onRefresh = new Signal();
        this.onNew = new Signal();
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
                e.visible = features.includes("refresh");
                e.onClick.connect(async () => {
                    this.enabled = false;

                    await postRequest("/temporal/storage/refresh", {
                        "store": store,
                    });

                    this.entries = await postRequest("/temporal/storage/list", {
                        "store": store,
                    });

                    this.enabled = true;

                    this.onRefresh.fire();
                });
            });

            e.createChild(ToolButton, (e) => {
                e.label = "\u{f15b}";
                e.visible = features.includes("new");
                e.onClick.connect(async () => {
                    this.enabled = false;

                    await postRequest("/temporal/storage/new", {
                        "store": store,
                    });

                    this.enabled = true;

                    this.onNew.fire();
                });
            });

            e.createChild(ToolButton, (e) => {
                e.label = "\u{f07c}";
                e.visible = features.includes("load");
                e.onClick.connect(async () => {
                    this.enabled = false;

                    let data = await postRequest("/temporal/storage/load", {
                        "store": store,
                        "name": this._dropdown.value,
                    });

                    this.enabled = true;

                    this.onLoad.fire(data);
                });
            });

            e.createChild(ToolButton, (e) => {
                e.label = "\u{f0c7}";
                e.visible = features.includes("save");
                e.onClick.connect(async () => {
                    if (!this.saveCallback) return;

                    let name = window.prompt("Enter name:", this._dropdown.value);
                    if (!name) return;

                    this.enabled = false;

                    await postRequest("/temporal/storage/save", {
                        "store": store,
                        "name": name,
                        "data": this.saveCallback(),
                    });

                    this.entries = await postRequest("/temporal/storage/list", {
                        "store": store,
                    });

                    this.enabled = true;

                    this.onSave.fire();
                });
            });

            e.createChild(ToolButton, (e) => {
                e.label = "\u{f303}";
                e.visible = features.includes("rename");
                e.onClick.connect(async () => {
                    let oldName = this._dropdown.value;
                    let newName = window.prompt("Enter new name:", oldName);
                    if (!newName) return;

                    this.enabled = false;

                    await postRequest("/temporal/storage/rename", {
                        "store": store,
                        "old_name": oldName,
                        "new_name": newName,
                    });

                    this.entries = await postRequest("/temporal/storage/list", {
                        "store": store,
                    });

                    this.enabled = true;

                    this.onRename.fire(oldName, newName);
                });
            });

            e.createChild(ToolButton, (e) => {
                e.label = "\u{f2ed}";
                e.visible = features.includes("delete");
                e.onClick.connect(async () => {
                    let name = this._dropdown.value;

                    if (!confirm(`Are you sure you want to delete "${name}"?`)) return;

                    this.enabled = false;

                    await postRequest("/temporal/storage/delete", {
                        "store": store,
                        "name": name,
                    });

                    this.entries = await postRequest("/temporal/storage/list", {
                        "store": store,
                    });

                    this.enabled = true;

                    this.onDelete.fire(name);
                });
            });
        });

        this.enabled = false;

        postRequest("/temporal/storage/list", {
            "store": store,
        }, (value) => {
            this.entries = value;
            this.enabled = true;
        });
    }

    get entries() {
        return this._dropdown.choices;
    }

    get value() {
        return this._dropdown.value;
    }

    set entries(value) {
        this._dropdown.choices = value;
    }

    set value(value) {
        this._dropdown.value = value;

        this.onValueChange.fire(value);
    }
}
customElements.define("fs-store-box", FSStoreBox);
