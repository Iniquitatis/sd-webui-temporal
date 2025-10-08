import {Accordion} from "/scripts/base/accordion.js";
import {Form} from "/scripts/base/form.js";
import {GroupBox} from "/scripts/base/group_box.js";
import {Tabs} from "/scripts/base/tabs.js";
import {FieldManager} from "/scripts/core/field_manager.js";
import {Signal} from "/scripts/core/signal.js";
import {defineElement} from "/scripts/utils/dom.js";
import {deepCopy} from "/scripts/utils/object.js";
import {getFieldDefinition} from "/scripts/object_field.js";
import {objectTypes} from "/scripts/shared_data.js";

export class ObjectForm extends Form {
    static tag = "ce-object-form";

    constructor(type, manager = null) {
        super(false);

        this.onValueChange = manager ? manager.onValueChange : new Signal();

        this._type = type;
        this._manager = manager ?? new FieldManager(this.onValueChange);
        if (!manager) this._manager.value.__type__ = type;
        this._lastTabs = null;
    }

    manage(key) {
        let field = objectTypes[this._type].fields[key];

        if (field.display == "accordion") {
            this._lastTabs = null;

            this.createChild(Accordion, (e) => {
                e.label = field.name;

                e.createChild(ObjectForm, (e) => {
                    e.manageAll();
                    e.value = deepCopy(field.default) ?? {};
                    this._manager.manage(e, key);
                }, field.type);
            });
        } else if (field.display == "group") {
            this._lastTabs = null;

            this.createChild(GroupBox, (e) => {
                e.label = field.name;

                e.createChild(ObjectForm, (e) => {
                    e.manageAll();
                    e.value = deepCopy(field.default) ?? {};
                    this._manager.manage(e, key);
                }, field.type);
            });
        } else if (field.display == "tab") {
            if (!this._lastTabs) {
                this._lastTabs = this.createChild(Tabs);
            }

            this._lastTabs.createTab(field.name, ObjectForm, (e) => {
                e.manageAll();
                e.value = deepCopy(field.default) ?? {};
                this._manager.manage(e, key);
            }, field.type);
        } else if (field.display == "unpack") {
            this._lastTabs = null;

            this.createChild(ObjectForm, (e) => {
                e.manageAll();
                e.value = deepCopy(field.default) ?? {};
                this._manager.manage(e, key);
            }, field.type);
        } else {
            this._lastTabs = null;

            let {cls, args, initializer, reader, writer} = getFieldDefinition(field);

            this.createField(field.name, cls, (e) => {
                initializer(e);
                this._manager.manage(e, key, reader, writer);

                if (field.dependencies) {
                    this.onValueChange.connect((value) => {
                        e.formItem.visible = areDependenciesSatisfied(value, field.dependencies);
                    });
                }
            }, ...(args ?? []));
        }
    }

    manageMultiple(keys) {
        for (let key of keys) {
            this.manage(key);
        }
    }

    manageAll() {
        for (let key of Object.keys(objectTypes[this._type].fields)) {
            this.manage(key);
        }
    }

    get value() {
        return this._manager.value;
    }

    set value(value) {
        this._manager.value = value;
    }
}
defineElement(ObjectForm);

function areDependenciesSatisfied(formValue, dependencies) {
    for (let [depKey, depValue] of Object.entries(dependencies)) {
        if (depValue instanceof Array) {
            if (!depValue.some((depChild) => formValue[depKey] == depChild)) {
                return false;
            }
        } else {
            if (formValue[depKey] != depValue) {
                return false;
            }
        }
    }

    return true;
}
